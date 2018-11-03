clear all; close all;

rng(1,'v4normal')

hold on;

NUM_ACCESS_POINTS = 5;
NUM_TRAINING_POINTS = 30;
NUM_TESTING_POINTS = 5;

AP = randn(NUM_ACCESS_POINTS, 2);
plot(AP(:,1), AP(:,2), 'k*')

Tr = randn(NUM_TRAINING_POINTS, 2);
plot(Tr(:,1), Tr(:,2), 'bo')

Te = randn(NUM_TESTING_POINTS, 2);
plot(Te(:,1), Te(:,2), 'ro')

% predictions = [ 
% ];
% 
% plot(predictions(:,1),predictions(:,2), 'r*')

legend([
    "Access Points"
    "Training Points"
    "Testing Points"
    "Predictions"
]);

hold off;


% Find the distances from all points AP by the training points
Tr_distance = zeros(length(Tr),NUM_ACCESS_POINTS);
for i = 1:length(Tr)
    Tr_distance(i,1:NUM_ACCESS_POINTS) = sqrt(sum(((AP - Tr(i,:)) .^ 2)'))';
end

% Find the distances from all points AP by the testing points
Te_distance = zeros(length(Te),NUM_ACCESS_POINTS);
for i = 1:length(Te)
    Te_distance(i,1:NUM_ACCESS_POINTS) = sqrt(sum(((AP - Te(i,:)) .^ 2)'))';
end

Tr_distance
Tr
Te_distance
Te

csvwrite('output/training_input.csv',Tr_distance)
csvwrite('output/training_output.csv',Tr)
csvwrite('output/testing_input.csv',Te_distance)
csvwrite('output/testing_output.csv',Te)

