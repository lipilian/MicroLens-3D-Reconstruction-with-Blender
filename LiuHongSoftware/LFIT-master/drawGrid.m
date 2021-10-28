function drawGrid(cal, newPath, filenames, batch)
    if batch
        outputPath = evalin('base','outputPath');
        savePath = fullfile(outputPath, 'PrecaliGrid');
        if ~exist(savePath, 'dir')
            mkdir(savePath);
        end
            
        for i = 1:length(filenames)
            img = imread(fullfile(newPath, filenames{i}));
            [r,c] = size(img);
            f = figure('visible', 'off');
            clf(f);
            imshow(img,'InitialMagnification',100);
            hold on;
            exactX_scatter = cal.exactX(:);
            exactY_scatter = cal.exactY(:);
            scatter(exactX_scatter, exactY_scatter,'r.');
            title('exact Lens Center(red) and round (blue)');
            hold on;
            roundX_Scatter = cal.roundX(:);
            roundY_Scatter = cal.roundY(:);
            scatter(roundX_Scatter, roundY_Scatter,'b.');
            
            uVectPad = evalin('base', 'uVectPad');
            vVectPad = evalin('base', 'vVectPad');
            
            for j = 1:length(exactX_scatter)
                hold on;
                xPixel = round(exactX_scatter(j)) - uVectPad;
                yPixel = round(exactY_scatter(j)) - vVectPad;
                minX = min(xPixel); maxX = max(xPixel);
                minY = min(yPixel); maxY = max(yPixel);
                rectangle('Position',[minX, minY, maxX - minX, maxY - minY], 'EdgeColor', 'g');
            end
            hold off;
            
            filename = filenames{i};
            filename_without_tail = split(filename, '.');
            filename_without_tail = filename_without_tail{1};
            %imwrite(f, fullfile(savePath,[filename_without_tail, '.tiff'])); 
           
            
            exportgraphics(f, fullfile(savePath,[filename_without_tail, '.tiff']), 'Resolution', 400);
            i
        end
          


    else
        %print('TODO: single image processing');
        %outputPath = evalin('base', 'outputPath');
        %savePath = fullfile(outputPath, 'SingleImgPrecaliGrid');
        %if ~exist(savePath, 'dir')
           % mkdir(savePath);
        %end
            


    end



end