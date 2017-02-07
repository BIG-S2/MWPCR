function [Mask_Idx, Mask_Loc,Nbr_Dist_2] = Head_File_For_Mask_MWPCR(mask,r)
% Generate the head file for the mask
% Input:
% mask: the 0/1 masking matrix
% r: the neighborhood range, defalt is 1; 1 step neighborhood
% Output:
% Mask_Idx: a matrix with first column being the index of the masked
% pixels/voxels; The rest colums is the row number of the associated 
% neighboorhood pixels/voxels in this matirx.
% Mask_Loc: a matrix with 1,2, or 3 columns, being the cooridinates of the masked
% pixels/voxels; 
% Nbr_Dist_Sqr: the squared distance between the neighborhood and the accociated
% pixels/voxels,
if nargin<2
    r = 1;
end
L = ceil(r);
index0 = find(mask);
q = length(index0);

imagesize = size(mask);
Dim = length(imagesize);

[d1,d2,d3] = ind2sub(imagesize,index0);% location of in-mask region

switch Dim
    case 1
        Mask_Loc = d1;
        Boundary_Check = sum(mask([1:L,(end-L+1):end]))>0;
    case 2
        Mask_Loc = [d1,d2];
        Boundary_Check1 = sum(mask([1:L,(end-L+1):end],:))>0;
        Boundary_Check2 = sum(mask(:,[1:L,(end-L+1):end]))>0;
        Boundary_Check = sum(Boundary_Check1(:))+sum(Boundary_Check2(:));
    case 3
        Mask_Loc = [d1,d2,d3];
        Boundary_Check1 = sum(mask([1:L,(end-L+1):end],:,:))>0;
        Boundary_Check2 = sum(mask(:,[1:L,(end-L+1):end],:))>0;
        Boundary_Check3 = sum(mask(:,:,[1:L,(end-L+1):end]))>0;
        Boundary_Check = sum(Boundary_Check1(:))+sum(Boundary_Check2(:))...
            +sum(Boundary_Check3(:));
    otherwise
        error('Image size not supported')
end

[Rel_Loc,Nbr_Dist_2] = Nbr_Radio_Based(r,Dim);
Nbr_Size = size(Rel_Loc,1);
Rel_Loc = kron(Rel_Loc,ones(q,1));
Base_Loc = repmat(Mask_Loc,Nbr_Size,1);
Nbr_Loc = Base_Loc+Rel_Loc;

if Boundary_Check>0
    Ind1 = Nbr_Loc<1;
    Limit_Loc = repmat(imagesize,q*Nbr_Size,1);
    Ind2 = Nbr_Loc>Limit_Loc;
    
    Ind = sum(Ind1+Ind2,2)>0;
    Ind_Loc = repmat(Ind,1,Dim);
    Nbr_Loc(Ind_Loc) = 1;
    switch Dim
        case 1
            Nbr_Index = Nbr_Loc;
        case 2
            Nbr_Index = sub2ind(imagesize,Nbr_Loc(:,1),Nbr_Loc(:,2));
        case 3
            Nbr_Index = sub2ind(imagesize,Nbr_Loc(:,1),Nbr_Loc(:,2),Nbr_Loc(:,3));
        otherwise
            error('Image size not supported')
    end
    Nbr_Index(Ind) = 0;
    Nbr_Index = reshape(Nbr_Index,q,Nbr_Size);
else
    switch Dim
        case 1
            Nbr_Index = Nbr_Loc;
        case 2
            Nbr_Index = sub2ind(imagesize,Nbr_Loc(:,1),Nbr_Loc(:,2));
        case 3
            Nbr_Index = sub2ind(imagesize,Nbr_Loc(:,1),Nbr_Loc(:,2),Nbr_Loc(:,3));
        otherwise
            error('Image size not supported')
    end
    Nbr_Index = reshape(Nbr_Index,q,Nbr_Size);
end
[~,Nbr_Index] = ismember(Nbr_Index,index0);% Find the row number of the NBR in the table
[Nbr_Dist_2,Dist_Order] = sort(Nbr_Dist_2); % Sort the index of nerborhood by distance
Nbr_Index = Nbr_Index(:,Dist_Order);
Mask_Idx = [index0,Nbr_Index];