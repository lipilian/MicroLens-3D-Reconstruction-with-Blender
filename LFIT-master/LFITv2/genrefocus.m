function refocusedImageStack = genrefocus(q,radArray,sRange,tRange,outputPath,imageSpecificName)
%GENREFOCUS Generates a series of refocused images as defined by the request vector.

% Copyright (c) 2014-2016 Dr. Brian Thurow <thurow@auburn.edu>
%
% This file is part of the Light-Field Imaging Toolkit (LFIT), licensed
% under version 3 of the GNU General Public License. Refer to the included
% LICENSE or <http://www.gnu.org/licenses/> for the full text.

fprintf('\nGenerating refocused views...');
progress(0);
clear vidobj; vidobj = 0;

switch q.fZoom
    case 'legacy'
        nPlanes = length(q.fAlpha);
        refocusedImageStack = zeros( length(tRange)*q.stFactor, length(sRange)*q.stFactor, nPlanes, 'single' );
    case 'telecentric'
        nPlanes = length(q.fPlane);
        refocusedImageStack = zeros( length(q.fGridY), length(q.fGridX), nPlanes, 'single' );
end

for fIdx = 1:nPlanes % for each refocused
    
    switch q.fZoom
        case 'legacy'
            qi          = q;
            qi.fAlpha   = q.fAlpha(fIdx);
            refocusedImageStack(:,:,fIdx) = refocus(qi,radArray,sRange,tRange);
        case 'telecentric'
            qi          = q;
            qi.fPlane   = q.fPlane(fIdx);
            refocusedImageStack(:,:,fIdx) = refocus(qi,radArray,sRange,tRange);
    end
    % Timer logic
    progress(fIdx,nPlanes);
end%for

limsStack=[min(refocusedImageStack(:)) max(refocusedImageStack(:))];

fprintf('\nDisplaying and/or saving refocused views...');
for fIdx = 1:nPlanes
    
    refocusedImage = refocusedImageStack(:,:,fIdx);
    
    switch q.contrast
        case 'slice',      limsSlice=[min(refocusedImage(:)) max(refocusedImage(:))]; refocusedImage = ( refocusedImage - limsSlice(1) )/( limsSlice(2) - limsSlice(1) ); refocusedImage = imadjust(refocusedImage,[q.intensity]);
        case 'stack',      refocusedImage = ( refocusedImage - limsStack(1) )/( limsStack(2) - limsStack(1) );  refocusedImage = imadjust(refocusedImage,[q.intensity]);
        otherwise           % Nothing to do
    end
    
    switch q.fZoom
        case 'legacy',      key = 'alpha';  val = q.fAlpha(fIdx);
        case 'telecentric', key = 'z';  val = q.fPlane(fIdx);
    end
    
    if q.title % Title image?
        
        cF = figure;
        switch q.title
            case 'caption',     caption = q.caption;
            case 'annotation',  caption = sprintf( '%s = %g', key,val );
            case 'both',        caption = sprintf( '%s --- [%s = %g]', q.caption, key,val );
        end
        displayimage(refocusedImage,caption,q.colormap,q.background);
        
        frame = getframe(1);
        expImage = frame2im(frame);
        
    else % no title
        expImage = refocusedImage;
    end%if
    
    if q.display % Display image?
        
        if q.title
            % Image already displayed, nothing to do
        else
            try
                set(0, 'currentfigure', cF);  % make refocus figure current figure (in case user clicked on another)
            catch
                cF = figure;
                set(cF,'position', [0 0 q.stFactor*size(radArray,4) q.stFactor*size(radArray,3)]);
                set(0, 'currentfigure', cF);  % make refocusing figure current figure (in case user clicked on another)
            end
            displayimage(refocusedImage,'',q.colormap,q.background);
        end
        
        switch q.display % How fast?
            case 'slow',   pause;
            case 'fast',   drawnow;
        end
        
    else
        
    end%if
    
    if q.saveas % Save image?
        
        dout = fullfile(outputPath,'Refocus');
        if ~exist(dout,'dir'), mkdir(dout); end
        fname = sprintf( '%s_refocus_stSS%g_uvSS%g_%s%g', imageSpecificName, q.stFactor, q.uvFactor, key, val);

        switch q.saveas
            case 'bmp'
                fout = fullfile(dout,[fname '.bmp']);
                imwrite(gray2ind(expImage,256),colormap([q.colormap '(256)']),fout);
                
            case 'png'
                fout = fullfile(dout,[fname '.png']);
                imwrite(gray2ind(expImage,256),colormap([q.colormap '(256)']),fout,'png','BitDepth',8);
                
            case 'jpg'
                fout = fullfile(dout,[fname '.jpg']);
                imwrite(gray2ind(expImage,256),colormap([q.colormap '(256)']),fout,'jpg','Quality',90,'BitDepth',8);
                
            case 'png16'
                if ~q.title % write colormap with file if no caption; otherwise, it is implied
                    fout = fullfile(dout,[fname '_16bit.png']);
                    if strcmpi(q.colormap,'gray')
                        imwrite(expImage,fout,'png','BitDepth',16);
                    else
                        imwrite(ind2rgb(gray2ind(expImage,65536),colormap([q.colormap '(65536)'])),fout,'png','BitDepth',16);
                    end
                else
                    fprintf('\n');
                    warning('16-bit PNG export is not supported when captions are enabled. Image not exported.');
                end
                
            case 'tif16'
                if ~q.title % write colormap with file if no caption; otherwise, it is implied
                    fout = fullfile(dout,[fname '_16bit.tif']);
                    if strcmpi(q.colormap,'gray')
                        imwrite(gray2ind(expImage,65536),fout,'tif','compression','lzw');
                    else
                        imwrite(ind2rgb(gray2ind(expImage,65536),colormap([q.colormap '(65536)'])),fout,'tif','compression','lzw');
                    end
                else
                    fprintf('\n');
                    warning('16-bit TIFF export is not supported when captions are enabled. Image not exported.');
                end
                
            case 'gif'
                fname = sprintf( '%s_refocusAnim_stSS%g_uvSS%g', imageSpecificName, q.stFactor, q.uvFactor);
                fout = fullfile(dout,[fname '.gif']);
                gifwrite(im2frame(gray2ind(expImage,256),colormap([q.colormap '(256)'])),q.colormap,fout,1/q.framerate,fIdx); % filename, delay, frame index
                
            case 'avi'
                fname = sprintf( '%s_refocusAnim_stSS%g_uvSS%g', imageSpecificName, q.stFactor, q.uvFactor);
                fout = fullfile(dout,[fname '.avi']);
                vidobj = aviwrite(im2frame(gray2ind(expImage,256),colormap([q.colormap '(256)'])),q.colormap,q.codec,vidobj,fout,fIdx,q.quality,q.framerate,nPlanes);
                
            case 'mp4'
                fname = sprintf( '%s_refocusAnim_stSS%g_uvSS%g', imageSpecificName, q.stFactor, q.uvFactor);
                fout = fullfile(dout,[fname '.mp4']);
                vidobj = mp4write(im2frame(gray2ind(expImage,256),colormap([q.colormap '(256)'])),q.colormap,vidobj,fout,fIdx,q.quality,q.framerate,nPlanes);
                
            otherwise
                error('Incorrect setting of the save flag in the requestVector input variable to the genrefocus function.');
                
        end%switch
                
    end%if
    
end%for

try     set(cF,'WindowStyle','normal'); % release focus
catch   % the figure couldn't be set to normal
end

fprintf('\nRefocusing generation finished.\n');
