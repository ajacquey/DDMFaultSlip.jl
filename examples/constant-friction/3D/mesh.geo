x0 = 0.0;
y0 = 0.0;
lc = 0.02*R;

Point(1) = {x0, y0, 0.0, lc};
Point(2) = {x0+R, y0, 0.0, lc};
Point(3) = {x0, y0+R, 0.0, lc};
Point(4) = {x0-R, y0, 0.0, lc};
Point(5) = {x0, y0-R, 0.0, lc};

Point(6) = {x0+R/10, y0, 0.0, lc/2};
Point(7) = {x0, y0+R/10, 0.0, lc/2};
Point(8) = {x0-R/10, y0, 0.0, lc/2};
Point(9) = {x0, y0-R/10, 0.0, lc/2};

Circle(1) = {2, 1, 3};
Circle(2) = {3, 1, 4};
Circle(3) = {4, 1, 5};
Circle(4) = {5, 1, 2};

Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};
Point{6} In Surface{1};
Point{7} In Surface{1};
Point{8} In Surface{1};
Point{9} In Surface{1};
Physical Surface(1) = {1};