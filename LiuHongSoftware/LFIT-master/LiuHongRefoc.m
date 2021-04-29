    %% 
% *Prerun for settings.*

targetPath = OpticInfo.target_path;
calibPath = OpticInfo.calibration_path;
outputPath = OpticInfo.output_path;
sensorType = OpticInfo.sensor_type;

%% mainlens information
focLenMain = 200; % main focal lens in mm
rulerHeight = 10.3; % rulerHeight in mm


%% microlens array and sensor information
pixelPitch = OpticInfo.pixel_mm; % mm/n
focLenMicro = OpticInfo.MLA_F_mm; % microlens array focal length in mm
microPitch = OpticInfo.MLA_size_mm; % microlens pitch in mm
sensorHight = OpticInfo.sensorH_resolution * pixelPitch; % 10 mm
numMicroX = OpticInfo.num_Micro_X; % 80 lens along width
numMicroY = OpticInfo.num_Micro_Y; % 80 lens along height
dmin = OpticInfo.dmin_mm; % 20 / 20 / 20 mm actual 
dmax = OpticInfo.dmax_mm;
dnum = OpticInfo.dnum;
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
%% Check first image with calibration
figure()
img = imread(fullfile(newPath, firstImage));
imageData = im2double(img);
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

%% TODO create 4D intensity matirx radArray
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

% reshape the image into 4D radArray
fprintf('\n Reshaping image into microimage stack...');
progress(0);

imStack = zeros(cal.numS, cal.numT, length(uVectPad), length(vVectPad), 'single');

numelST = cal.numS * cal.numT;
showCase = randperm(numelST, 10);
figure()
imshow(img);
OriginalPoints = []; %  point of lens array center
directions = []; % direction of ray 
for k = 1:numelST
    [s,t] = ind2sub([cal.numS cal.numT], k);
    xPixel = round(cal.exactX(s,t)) - uVectPad;
    yPixel = round(cal.exactY(s,t)) - vVectPad;
    
    
    
    if ismember(k, showCase)
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
    
    imStack(s,t,:,:) = imageData(yPixel,xPixel).*mask;
    progress(k, numelST);
end
hold off;
title('10 random lens from the raw image to ensure 4d radArray correction');

directions = normr(directions);
% convert origion data from pixel units to mm units

%%
xy = calIntersect(OriginalPoints, directions, 70);
figure()
scatter(xy(:,1),xy(:,2),'.')

%% histogram count 
scaleFactor = 1;
minX = -2; maxX = 20; NumX = round((maxX - minX)/OpticInfo.MLA_size_mm * scaleFactor);
minY = -2; maxY = 12; NumY = round((maxY - minY)/OpticInfo.MLA_size_mm * scaleFactor);
minZ = -100; maxZ = 100; NumZ = 400;
Xedge = linspace(minX,maxX,NumX);
Xcenter = movmean(Xedge,2); Xcenter = Xcenter(2:end);
Yedge = linspace(minY,maxY,NumY);
Ycenter = movmean(Yedge,2); Ycenter = Ycenter(2:end);
Yedge = linspace(minY,maxY,NumY);
RayCounts = zeros(NumX-1, NumY-1, NumZ-1);


Zcenter = linspace(minZ,maxZ,NumZ);
for i = 1:NumZ
    xy = calIntersect(OriginalPoints,directions,Zcenter(i));
    N = histcounts2(xy(:,1),xy(:,2),Xedge,Yedge);
    RayCounts(:,:,i) = N;
end

%%
[X,Y,Z] = ndgrid(Xcenter,Ycenter,Zcenter);
maxCount = 100;
xx = X(RayCounts > maxCount); yy = Y(RayCounts > maxCount); zz = Z(RayCounts > maxCount);
color = RayCounts(RayCounts> maxCount); 
figure()
xyz = [xx,yy,zz];
pcshow(xyz, color);
caxis([maxCount, 200]);




% xy = calIntersect(OriginalPoints, directions, 70);
% [N,~,~] = histcounts2(xy(:,1),xy(:,2),Xedge,Yedge);
% [X,Y] = meshgrid(Xcenter,Ycenter);
% figure()
% [C,h] = contourf(X,Y,N',30);
% caxis([0,30])
% set(h,'LineColor','none')
% colormap(jet);




%%
function xy0 = calIntersect(xy0, directions, d)
    [M, ~] = size(xy0);
    d = repmat(d, M, 1);
    factor = d./directions(:,3); 
    xy0(:,1) = xy0(:,1) + factor.*directions(:,1);
    xy0(:,2) = xy0(:,2) + factor.*directions(:,2);



end
