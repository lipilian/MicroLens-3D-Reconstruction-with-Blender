    %% 
% *Prerun for settings.*

targetPath = 'C:\Users\liuhong2\Desktop\MicroLens-3D-Reconstruction-with-Blender\LFIT_test\MicroscopeTest\target';
calibPath = 'C:\Users\liuhong2\Desktop\MicroLens-3D-Reconstruction-with-Blender\LFIT_test\MicroscopeTest\cali';
outputPath = 'C:\Users\liuhong2\Desktop\MicroLens-3D-Reconstruction-with-Blender\LFIT_test\MicroscopeTest\output';
sensorType = 'rect';

%% mainlens information
focLenMain = 100; % main focal lens in mm
rulerHeight = 10.3; % rulerHeight in mm


%% microlens array and sensor information
pixelPitch = 0.00435; % mm/n
focLenMicro = 3.75; % microlens array focal length in mm
microPitch = 0.125; % microlens pitch in mm
sensorHight = 2400 * 0.00435; % 10 mm
numMicroX = 130; % 80 lens along width
numMicroY = 80; % 80 lens along height
dmin = -20; % 20 / 20 / 20 mm actual 
dmax = 20;
dnum = 100;
dplane = linspace(dmin,dmax,dnum);

microDiameterExact = microPitch/pixelPitch % diameter of a microlens in pixels/n

magnification = -sensorHight/rulerHeight; % magnigication 

numThreads = 2;

% TODO: not sure if it will be useful
si = focLenMain*(1-magnification);
sizePixelAperture = (si*pixelPitch)/focLenMicro;

% check if need calibration
calibration = true;

% set Load and save factor 
Mode = 'tagAutoLoadSave'
switch Mode
    case 'tagNoLoadSave'
        loadFlag = 0;
        saveFlag = 0;
    case 'tagAutoLoadSave'
        loadFlag = 1;
        saveFlag = 1;
    case 'tagClearCalSaveNew'
        loadFlag = 2;
        saveFlag = 1;
    case 'tagLoadExternalCal'
        loadFlag = 3;
        saveFlag = 0;
end

imageSetName = 'Test';

% Single Image Mode
%% calibrate the image
numImages           = 1;
imageIndex          = 1;
refocusedImageStack = 0;
fprintf('LFI_Toolkit Demonstration Program\n');
fprintf('-------------------------------------------------\n');
[LFI_path,progVersion] = toolkitpathv2();
if calibration
    calImagePath = imageavg(calibPath,'avgcal.tif');
else
    calImagePath = 0; % does not matter, we will load external calibration file
end
cal = computecaldata(calibPath,...
                    calImagePath, ...
                    loadFlag, ...
                    saveFlag, ...
                    imageSetName, ...
                    sensorType, ...
                    numMicroX, ...
                    numMicroY, ...
                    microPitch, ...
                    pixelPitch)
%% Load first image
[firstImage,newPath] = uigetfile({'*.tiff; *.tif','TIFF files (*.tiff, *.tif)'},'Select a single raw plenoptic image to begin processing...',targetPath);
newPath = newPath(1:end-1); % removes trailing slash from path
if firstImage == 0
    % Program is over. User did NOT select a file to import.
    warning('File not selected for import. Program execution ended.');                
else
    imageName = struct('name',firstImage);
            
    % Update variables
    imageSpecificName = [imageSetName '_' imageName(imageIndex).name(1:end-4)]; %end-4 removes .tif
    imagePath = fullfile(targetPath,imageName(imageIndex).name);
    
    % Interpolate image data
    [radArray,sRange,tRange] = interpimage2(cal,imagePath,sensorType,microPitch,pixelPitch,numMicroX,numMicroY);
    % radArray: 4D intensity matrix (u, v, s, t);
    
    % Typecast variables as single to conserve memory
    radArray        = single(radArray);
    sRange          = single(sRange);
    tRange          = single(tRange);
    interpPadding   = single(1); % hard coded; if the padding in interpimage2.m changes, change this accordingly.
    microRadius     = single( floor(size(radArray,1)/2) - interpPadding ); % since we've padded the extracted data by a pixel in interpimage2, subtract 1
    
    % microscope use 'telecentric' mode:
    SS_UV = 1; SS_ST = 1; % no supersampling
    % 'rect' microLens array not need for mask
    mask = ones(1+2*SS_UV*microRadius);
    
    % Create (u,v) and (s,t) arrays
    sizeS       = length(sRange)*SS_ST;
    sizeT       = length(tRange)*SS_ST;

    uRange      = linspace( microRadius, -microRadius, 1+2*microRadius );
    vRange      = linspace( microRadius, -microRadius, 1+2*microRadius );

    uSSRange    = linspace( microRadius, -microRadius, 1+2*SS_UV*microRadius );
    vSSRange    = linspace( microRadius, -microRadius, 1+2*SS_UV*microRadius );
    sSSRange    = linspace( sRange(1), sRange(end), sizeS );
    tSSRange    = linspace( tRange(1), tRange(end), sizeT );
    
    % Memory preallocation
    GridX = length(sRange);
    GridY = length(tRange);
    
%     imageProduct    = ones( length(GridY), length(GridX), 'single' );
    %imageIntegral   = zeros( length(GridY), length(GridX), 'single' );
%     filterMatrix    = zeros( length(GridY), length(GridX), 'single' );
%     
    % Crop and reshape 4D-array to optimize parfor performance
        % TODO: Check if crop is necessary
    % radArray = radArray( 1+interpPadding:end-interpPadding, 1+interpPadding:end-interpPadding, :, : );
    radArray = permute( radArray, [2 1 4 3] );
    radArray = reshape( radArray, size(radArray,1)*size(radArray,2), size(radArray,3), size(radArray,4) );

    % TODO: no need to care about the sizePixelAperature
    [sActual,tActual] = meshgrid( sRange, tRange );
    [uActual,vActual] = meshgrid( uRange*pixelPitch, vRange*pixelPitch );

    numelUV = numel(uActual);
    refocusedImageStack = zeros(GridY,GridX,length(dplane));
    for k = 1:length(dplane)
        d = dplane(k);
        imageIntegral   = zeros(GridY,GridX, 'single' );
        parfor( uvIdx = 1:numelUV, Inf)
            sp_p = 0; tp_p = 0;
            if mask(uvIdx) > 0
                sp_p = d / focLenMicro * ...
                    uActual(uvIdx) + ...
                    (1 - d / focLenMicro) * ...
                    sActual;
                tp_p = d / focLenMicro * ...
                    vActual(uvIdx) + ...
                    (1 - d / focLenMicro) * ...
                    tActual;
            end
            extractedImageTemp = interp2(sp_p, tp_p, squeeze(radArray(uvIdx,:,:)), sActual, tActual, '*linear' ,0);

            imageIntegral = imageIntegral + extractedImageTemp;
        end
        syntheticImage = imageIntegral;
        syntheticImage(syntheticImage<0) = 0;
        refocusedImageStack(:,:,k) = syntheticImage;
    end
    limsStack=[min(refocusedImageStack(:)) max(refocusedImageStack(:))];
    
    fprintf('\nDisplaying and/or saving refocused views...');
    clear vidobj; vidobj = 0;
    for fIdx = 1:length(dplane)
        refocusedImage = refocusedImageStack(:,:,fIdx);
        limsSlice=[min(refocusedImage(:)) max(refocusedImage(:))]; 
        refocusedImage = ( refocusedImage - limsSlice(1) )/( limsSlice(2) - limsSlice(1) ); 
        refocusedImage = imadjust(refocusedImage,[0 1]);
        key = 'd';
        val = dplane(fIdx);
        
        expImage = refocusedImage;
        
        dout = fullfile(outputPath,'Refocus');
        if ~exist(dout,'dir'), mkdir(dout); end
        fname = sprintf( 'refocus_%s%g', key, val);
        fout = fullfile(dout,[fname '.avi']);
        vidobj = aviwrite(im2frame(gray2ind(expImage,256),colormap(['gray' '(256)'])),'gray','uncompressed',vidobj,fout,fIdx,100,1,length(dplane));
    end
end

%%
