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

% m is the number of examples
m = size(X, 1)

pause

for i = 1:m
    % Get each sample from X
    sample_temp = X(i, :)
    distance_min = 0;

    for j = 1:K
        % Get each centroid
        centroid_temp = centroids(j, :)

        % Get the distance between example point and centroid
        distance_temp = sum( (sample_temp - centroid_temp).^2 )

        % reset the distance_min as distance_temp for 1st inner loop
        if distance_min == 0
            distance_min = distance_temp
            idx(i) = j;
        end

        % Assign the minimum distance value to distance_min
        if distance_temp < distance_min
            distance_min = distance_temp
            idx(i) = j;
        end
    end
end

% =============================================================

end

