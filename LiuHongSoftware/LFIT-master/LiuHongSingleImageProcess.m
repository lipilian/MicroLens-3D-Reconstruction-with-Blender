
function [X,Y,Z,RayCounts] = LiuHongSingleImageProcess(OpticInfo, NumF, calibration, batch)
    % ------------------------------------------------------
    % OpticInfo include all optic information and output resolution
    % information 
    % NumF which image user want to processed 
    % ------------------------------------------------------
    
    targetPath = OpticInfo.target_path;
    calibPath = OpticInfo.calibration_path;
    outputPath = OpticInfo.output_path;
    sensorType = OpticInfo.sensor_type;
    
    pixelPitch = OpticInfo.pixel_mm; % mm/n
    focLenMicro = OpticInfo.MLA_F_mm; % microlens array focal length in mm
    microPitch = OpticInfo.MLA_size_mm; % microlens pitch in mm
    sensorHight = OpticInfo.sensorH_resolution * pixelPitch; % 10 mm
    numMicroX = OpticInfo.num_Micro_X; % 80 lens along width
    numMicroY = OpticInfo.num_Micro_Y; % 80 lens along height
    
    minIntensity = OpticInfo.minIntensity;
    maxRayCount = OpticInfo.maxRayCount;

    scaleFactor = OpticInfo.scaleFactor;
    minX = OpticInfo.xmin_mm; maxX = OpticInfo.xmax_mm; NumX = round((maxX - minX)/OpticInfo.MLA_size_mm * scaleFactor);
    minY = OpticInfo.ymin_mm; maxY = OpticInfo.ymax_mm; NumY = round((maxY - minY)/OpticInfo.MLA_size_mm * scaleFactor);
    minZ = OpticInfo.dmin_mm; maxZ = OpticInfo.dmax_mm; NumZ = OpticInfo.dnum;
    
    
    Xedge = linspace(minX,maxX,NumX);
    Xcenter = movmean(Xedge,2); Xcenter = Xcenter(2:end);
    Yedge = linspace(minY,maxY,NumY);
    Ycenter = movmean(Yedge,2); Ycenter = Ycenter(2:end);
    Yedge = linspace(minY,maxY,NumY);
    RayCounts = zeros(NumX-1, NumY-1, NumZ-1);
    Zcenter = linspace(minZ,maxZ,NumZ);


    Mode = 'tagAutoLoadSave';
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
    % find all images name and their path
    
    
    
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
                    pixelPitch);
    
    microRadius = floor((microPitch/pixelPitch)/2); % (-14 to 14)
    microPad = 1;

    uVect = single(microRadius : -1 : -microRadius);
    vVect = single(microRadius : -1 : -microRadius);

    [v,u] = ndgrid(vVect, uVect);

    % Define (u,v) coordinate vectors with padding
    uVectPad = single(microRadius + microPad : -1 : -microRadius - microPad);
    vVectPad = single(microRadius + microPad : -1 : -microRadius - microPad);

    % define mask, since this is 'rect' type lens, we use full aperture
    mask = ones( 1 + 2*(microRadius + microPad)); % (31 by 31)
                
                
                
    if calibration && ~batch % only if calibration is not performed yet
        [firstImage,newPath] = uigetfile({'*.tiff; *.tif','TIFF files (*.tiff, *.tif)'},'Select a single raw plenoptic image to begin processing...',targetPath);
        newPath = newPath(1:end-1); % removes trailing slash from 
        img = imread(fullfile(newPath, firstImage));
        % TODO: implement the single image draw grid function
        % inside of drawGrid function
    elseif calibration && batch
        fileNames = getImagePathList(targetPath);
        newPath = targetPath;
        drawGrid(cal, newPath, fileNames, batch);
    elseif batch
        fileNames = getImagePathList(targetPath);
        newPath = targetPath;
    end
    
    fprintf('\n Reshaping image into microimage stack... \n');
    numelST = cal.numS * cal.numT;
    
    %OriginalPointsCell = {};
    %directionsCell = {};

    if batch
        RayCounts = zeros(NumX-1, NumY-1, NumZ-1, length(fileNames));
        for i = 1:length(fileNames)
            img = imread(fullfile(newPath, fileNames{i}));
            [m,n] = size(img);
            OriginalPoints = [];
            directions = [];
            for k = 1:numelST
                [s,t] = ind2sub( [cal.numS cal.numT], k );
                xPixel = round(cal.exactX(s,t)) - uVectPad; 
                yPixel = round(cal.exactY(s,t)) - vVectPad;
                minX = min(xPixel); maxX = max(xPixel);
                minY = min(yPixel); maxY = max(yPixel);
                if minY < 1 || maxY > m
                    continue;
                end
                if minX < 1 || maxX > n
                    continue;
                end
                subImage = img(yPixel, xPixel);
                [yID, xID] = find(subImage >= OpticInfo.minIntensity);
                xRay = xPixel(xID) * pixelPitch;
                yRay = yPixel(yID) * pixelPitch;
                L = length(xRay); zRay = ones(L,1) * OpticInfo.MLA_F_mm;
                xLen = cal.exactX(s,t) * pixelPitch;
                yLen = cal.exactY(s,t) * pixelPitch;
                xyLen = repmat([xLen, yLen], L, 1);
                OriginalPoints = [OriginalPoints; xyLen];
                tempDirection = [[xRay', yRay'] - xyLen, zRay];
                directions = [directions; tempDirection];
            end
            %OriginalPointsCell{i} = OriginalPoints; %Save the unit vector information frame by frame
            directions = normr(directions);
            %directionsCell{i} = directions; % same as above
            for j = 1:NumZ
                xy = calIntersect(OriginalPoints,directions,Zcenter(j));
                N = histcounts2(xy(:,1),xy(:,2),Xedge,Yedge);
                RayCounts(:,:,j,i) = N;
            end
            
        end
         
    else
        %TODO: implement the single image 3D reconstrcution
    end
    [X,Y,Z] = ndgrid(Xcenter,Ycenter,Zcenter);

%     for i = 1:NumZ
%         xy = calIntersect(OriginalPoints,directions,Zcenter(i));
%         N = histcounts2(xy(:,1),xy(:,2),Xedge,Yedge);
%         RayCounts(:,:,i) = N;
%     end
    
%     [X,Y,Z] = ndgrid(Xcenter,Ycenter,Zcenter);
%     if calibration
%         xx = X(RayCounts > maxRayCount); yy = Y(RayCounts > maxRayCount); zz = Z(RayCounts > maxRayCount);        
%         color = RayCounts(RayCounts> maxRayCount); 
%         figure()
%         xyz = [xx,yy,zz];
%         pcshow(xyz, color);
%         caxis([maxRayCount, 200]);
%     end
     
end
    
