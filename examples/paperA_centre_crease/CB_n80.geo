// Centre-crease benchmark: [-1, 1]^2, crease (-0.5, 0) -> (0.5, 0) on the y = 0 node row.
SetFactory("Built-in");
Mesh.SaveAll = 0;
Point(1) = {-1, -1, 0}; Point(2) = {1, -1, 0}; Point(3) = {1, 1, 0}; Point(4) = {-1, 1, 0};
Line(1) = {1, 2}; Line(2) = {2, 3}; Line(3) = {3, 4}; Line(4) = {4, 1};
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};
Transfinite Curve {1, 2, 3, 4} = 81;
Transfinite Surface {1};
Recombine Surface {1};

// Crease: its own line, not embedded in the sheet; one segment per element.
Point(10) = {-0.5, 0, 0}; Point(11) = {0.5, 0, 0};
Line(5) = {10, 11};
Transfinite Curve {5} = 41;

Physical Surface("Domain", 100) = {1};
Physical Curve("Crease", 5) = {5};
Physical Point(1) = {1};
Physical Point(2) = {2};
Physical Point(4) = {4};
