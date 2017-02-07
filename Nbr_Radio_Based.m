function [Loc_Rel,Dist_Sqr] = Nbr_Radio_Based (r,Dim)
% Find the neighborhood with s.t. ||x-x_0||^2<=r^2
% Input:
% r = the maximum distance from the center point;
% Dim: dimension of the images, can take value of 1,2 or 3.
% Output:
% Loc_Rel: the relative location of the NBR's;
% Dist_Sqr: the associated suqared distance of the NBR's.
Loc_Rel = [];
Dist_Sqr = [];
L = ceil(r);
switch Dim
    case 1
        for d1 = -L:L
            Dist = d1^2;
            if Dist<=r^2 && d1
                Loc_Rel = [Loc_Rel; d1];
                Dist_Sqr = [Dist_Sqr,Dist];
            end
        end
    case 2
        for d1 = -L:L
            for d2 = -L:L
                Dist = d1^2+d2^2;
                if Dist<=r^2 && (d1^2 + d2^2)
                    Loc_Rel = [Loc_Rel; [d1,d2]];
                    Dist_Sqr = [Dist_Sqr,Dist];
                end
            end
        end
    case 3
        for d1 = -L:L
            for d2 = -L:L
                for d3 = -L:L
                    Dist = d1^2+d2^2+d3^2;
                    if Dist<=r^2 && (d1^2 + d2^2 + d3^2)
                        Loc_Rel = [Loc_Rel; [d1,d2,d3]];
                        Dist_Sqr = [Dist_Sqr,Dist];
                    end
                end
            end
        end
    otherwise
        error('Image size not supported!')
end