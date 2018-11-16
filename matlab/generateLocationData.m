clear all; close all;

rng(1,'v4normal')

hold on;

NUM_ACCESS_POINTS = 5;
NUM_TRAINING_POINTS = 100000;
NUM_TESTING_POINTS = 500;

AP = randn(NUM_ACCESS_POINTS, 2);
plot(AP(:,1), AP(:,2), 'k*')

Tr = rand(NUM_TRAINING_POINTS, 2) * 2 - 1;
% plot(Tr(:,1), Tr(:,2), 'bo')

Te = rand(NUM_TESTING_POINTS, 2) * 2 - 1;
plot(Te(:,1), Te(:,2), 'ro')

predictions = [

 -0.7354056030149186,  0.6796921612680191
 -0.5485747869029229,  0.5536379585814766
 0.05949042425421569,  0.5958815663226174
  0.7978962232878265,-0.15222730033141085
 -0.1039848647117072, 0.12351144678512038
-0.23181591620332898,  0.6541029396842242
 -0.6038202965383983,  0.4111107560442522
 -0.8533249617939161, -0.9749371872125391
0.050557658299523855,  0.7581062984481821
 -0.6225122711415513,  0.7494532862928669
  0.2779164540977387,  0.6886432466095456
  -0.794991649431388, 0.27794064976868404
  0.5581249721494693,-0.02949296324362
 -0.0699204368658437,  -0.901380739086489
 -0.6388140743700617, 0.31924239206449534
 -0.7431709390439349, 0.07724835611214553
 -0.1735391283378115,  0.7257696573456538
 -0.3660780871744709, -0.1193602180718466
 -0.6801315096908898,  0.6437378462366559
-0.21544985347361179,  0.4971738195033575

];

plot(predictions(:,1),predictions(:,2), 'r*')

% Add labels to understand visually how far off we were!
k = 1:min(length(Te),length(predictions));
text(predictions(:,1),predictions(:,2),num2str(k'), 'Color', 'blue')
text(Te(k,1),Te(k,2),num2str(k'))

for k = 1:min(length(Te),length(predictions))
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

