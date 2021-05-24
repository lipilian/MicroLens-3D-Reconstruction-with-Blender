function xy0 = calIntersect(xy0, directions, d)
    [M, ~] = size(xy0);
    d = repmat(d, M, 1);
    factor = d./directions(:,3); 
    xy0(:,1) = xy0(:,1) + factor.*directions(:,1);
    xy0(:,2) = xy0(:,2) + factor.*directions(:,2);
end   