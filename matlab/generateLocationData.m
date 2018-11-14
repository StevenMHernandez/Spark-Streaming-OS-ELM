clear all; close all;

rng(1,'v4normal')

hold on;

NUM_ACCESS_POINTS = 5;
NUM_TRAINING_POINTS = 100000;
NUM_TESTING_POINTS = 5;

AP = randn(NUM_ACCESS_POINTS, 2);
plot(AP(:,1), AP(:,2), 'k*')

Tr = randn(NUM_TRAINING_POINTS, 2);
% plot(Tr(:,1), Tr(:,2), 'bo')

Te = randn(NUM_TESTING_POINTS, 2);
plot(Te(:,1), Te(:,2), 'ro')

predictions = [
    1.1887117768793072 -1.1037663821927928
    -0.8044335008140409 -1.7614652520939118
    -0.6223433172987565 0.44935435383664313
    0.6280652212130553 0.4791563135308561
    0.17514932897473212 0.6562785029690827
];

plot(predictions(:,1),predictions(:,2), 'r*')

% Add labels to understand visually how far off we were!
k = 1:length(Te);
text(predictions(:,1),predictions(:,2),num2str(k'), 'Color', 'blue')
text(Te(:,1),Te(:,2),num2str(k'))

for k = 1:length(Te)
    a = Te(k,:)
    b = predictions(k,:)
    line([a(1) b(1)], [a(2) b(2)], 'Color', 'cyan')
end

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

Te_distance
Te

csvwrite('output/train_XY.csv',[Tr_distance Tr])
csvwrite('output/test_XY.csv', [Te_distance Te])

