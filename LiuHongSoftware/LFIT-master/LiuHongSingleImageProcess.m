
function [X,Y,Z,RayCounts] = LiuHongSingleImageProcess(OpticInfo, NumF, calibration)
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
                    pixelPitch);
             
    if calibration % only if calibration is not performed yet
        [firstImage,newPath] = uigetfile({'*.tiff; *.tif','TIFF files (*.tiff, *.tif)'},'Select a single raw plenoptic image to begin processing...',targetPath);
        newPath = newPath(1:end-1); % removes trailing slash from path
    else
        fileList = dir(targetPath);
        fileNames = extractfield(fileList,'name');
        fileName = fileNames{NumF + 3};
        newPath = targetPath;
    end
    
    if calibration
        img = imread(fullfile(newPath, firstImage));
        figure()
        imshow(img);
        hold on;
        exactX_Scatter = cal.exactX(:);
        exactY_Scatter = cal.exactY(:);
        scatter(exactX_Scatter, exactY_Scatter, 'r.');
        hold off;
        title('exact Lens Center');
        figure()
        imshow(img);
        hold on;
        roundX_Scatter = cal.roundX(:);
        roundY_Scatter = cal.roundY(:);
        scatter(roundX_Scatter, roundY_Scatter, 'b.');
        hold off;
        title('round Lens Center');
    else
        img = imread(fullfile(newPath, fileName));
    end
    
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

    fprintf('\n Reshaping image into microimage stack...');
    
    numelST = cal.numS * cal.numT;
    
    if calibration
        showCase = randperm(numelST, 10);
        figure();
        imshow(img);   
    end
    
    OriginalPoints = [];
    directions = [];
    
    for k = 1:numelST 
        [s,t] = ind2sub([cal.numS cal.numT], k);
        xPixel = round(cal.exactX(s,t)) - uVectPad;
        yPixel  = round(cal.exactY(s,t)) - vVectPad;

        
        if  calibration && ismember(k, showCase)
            hold on;
            plot(cal.exactX(s,t), cal.exactY(s,t), 'r*');
            minX = min(xPixel); maxX = max(xPixel);
            minY = min(yPixel); maxY = max(yPixel);
            hold on;
            rectangle('Position',[minX, minY, maxX - minX, maxY - minY],'EdgeColor','b');
        end

        subImage = img(yPixel, xPixel);
        [yID, xID] = find(subImage == 255);
        xRay = xPixel(xID) * pixelPitch; 
        yRay = yPixel(yID) * pixelPitch;
        L = length(xRay); zRay = ones(L,1) * OpticInfo.MLA_F_mm;
        
        xLen = cal.exactX(s,t) * pixelPitch;
        yLen = cal.exactY(s,t) * pixelPitch;
        xyLen = repmat([xLen, yLen],L,1);
        OriginalPoints = [OriginalPoints; xyLen];
        tempDirection = [[xRay', yRay'] - xyLen, zRay];
        directions = [directions; tempDirection];
    end

    if calibration
        hold off;
        title('10 random lens from the raw image to ensure 4d radArray correction');
    end
    
    directions = normr(directions);
    xy = calIntersect(OriginalPoints, directions, 30);

    if calibration
        figure()
        scatter(xy(:,1),xy(:,2),'.');
        title('ray intersection point on d = 30 mm');
    end

    for i = 1:NumZ
        xy = calIntersect(OriginalPoints,directions,Zcenter(i));
        N = histcounts2(xy(:,1),xy(:,2),Xedge,Yedge);
        RayCounts(:,:,i) = N;
    end
    
    [X,Y,Z] = ndgrid(Xcenter,Ycenter,Zcenter);
    if calibration
        xx = X(RayCounts > maxRayCount); yy = Y(RayCounts > maxRayCount); zz = Z(RayCounts > maxRayCount);        
        color = RayCounts(RayCounts> maxRayCount); 
        figure()
        xyz = [xx,yy,zz];
        pcshow(xyz, color);
        caxis([maxRayCount, 200]);
    end
     
end
    
function xy0 = calIntersect(xy0, directions, d)
    [M, ~] = size(xy0);
    d = repmat(d, M, 1);
    factor = d./directions(:,3); 
    xy0(:,1) = xy0(:,1) + factor.*directions(:,1);
    xy0(:,2) = xy0(:,2) + factor.*directions(:,2);
end   