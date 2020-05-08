function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%






# X [m 2] c [3 2]
#defining our column centroid norm matrix
centroid_norm = zeros(size(X,1),K);
# centroid_norm  = column sum of [X - c(1:K,:)].^2
for i=1:K
  centroid_norm(:,i) = sum((X-centroids(i,:)).^2,2);
endfor

# pick min column index of centroid_norm matrix which will be your minimum distance center
[val  idx] =min(centroid_norm,[],2);

##for i=1:size(X,1)
##  idx(i) = centroids(column_index(i),:)  
##endfor



% =============================================================

end

