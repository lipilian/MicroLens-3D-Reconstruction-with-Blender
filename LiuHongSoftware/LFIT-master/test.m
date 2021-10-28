
for d = 1:20:400
    figure(1)
    clf;
    
    
    %scatter(xy(:,1),xy(:,2),'.');
    figure(1)
    clf;
    N = RayCounts(:,:,300,1);
    N(N < 40) = 0;
    [C,h] = contourf(N,100);
    set(h,'LineColor','none');
    colormap('jet');
    caxis([0 200]);
    axis equal;
    
    figure(2)
    clf;
    xy = calIntersect(OriginalPoints,directions,20);
    N = histcounts2(xy(:,1),xy(:,2),Xedge,Yedge);
    N(N< 40) = 0;
    [C,h] = contourf(N,100);
    set(h,'LineColor','none');
    colormap('jet');
    caxis([0 200]);
    axis equal;
    
end


% FrameID = 0
% RayCountsByFrame = RayCounts(:,:,:,1);
% maxRayCount = 10;
% xx = X(RayCountsByFrame > maxRayCount); yy = Y(RayCountsByFrame > maxRayCount); zz = Z(RayCountsByFrame > maxRayCount);        
% color = RayCounts(RayCountsByFrame > maxRayCount); 
% figure()
% xyz = [xx,yy,zz];
% pcshow(xyz, color);
% caxis([maxRayCount, 100]);




