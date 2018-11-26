clear all; close all;

rng(1,'v4normal')

hold on;

NUM_ACCESS_POINTS = 4;

NUM_SAMPLES_PER_TRAINING_POINT = 200;

NUM_TRAINING_POINTS_1 = 10;
NUM_TRAINING_POINTS_2 = 5;
NUM_TRAINING_POINTS_3 = 5;
NUM_TESTING_POINTS_1 = 5;
NUM_TESTING_POINTS_2 = 5;
NUM_TESTING_POINTS_3 = 5;

AP = [
    0 0
    20 0
    0 20
    20 20
];

k = NUM_SAMPLES_PER_TRAINING_POINT
[Tr1_signals, Tr1_actualSmall, Tr1_actual] = makeNWithKSamples(NUM_TRAINING_POINTS_1, k, 2.0, AP);
[Tr2_signals, Tr2_actualSmall, Tr2_actual] = makeNWithKSamples(NUM_TRAINING_POINTS_2, k, 2.0, AP);
[Tr3_signals, Tr3_actualSmall, Tr3_actual] = makeNWithKSamples(NUM_TESTING_POINTS_3, k, 2.0, AP);
[Te1_signals, Te1_actualSmall, Te1_actual] = makeNWithKSamples(NUM_TESTING_POINTS_1, k, 2.0, AP);
[Te2_signals, Te2_actualSmall, Te2_actual] = makeNWithKSamples(NUM_TESTING_POINTS_2, k, 2.0, AP);
[Te3_signals, Te3_actualSmall, Te3_actual] = makeNWithKSamples(NUM_TESTING_POINTS_3, k, 2.0, AP);

APplots = plot(AP(:,1), AP(:,2), 'k*');
TrainingPlotsOne = plot(Tr1_actualSmall(:,1), Tr1_actualSmall(:,2), 'o');
TrainingPlotsTwo = plot(Tr2_actualSmall(:,1), Tr2_actualSmall(:,2), 'o');
TrainingPlotsThree = plot(Tr3_actualSmall(:,1), Tr3_actualSmall(:,2), 'o');
TestingPlotsOne = plot(Te1_actualSmall(:,1), Te1_actualSmall(:,2), '^');
TestingPlotsTwo = plot(Te2_actualSmall(:,1), Te2_actualSmall(:,2), '^');
TestingPlotsThree = plot(Te3_actualSmall(:,1), Te3_actualSmall(:,2), '^');
plot(AP(:,1), AP(:,2), 'k*'); % Again to plot it on top!

% Add labels to understand visually how far off we were!
% k = 1:min(length(Te),length(predictions));
% text(predictions(:,1),predictions(:,2),num2str(k'), 'Color', 'blue')
% text(Te(k,1),Te(k,2),num2str(k'))

% for k = 1:min(length(Te),length(predictions))
%     a = Te(k,:)
%     b = predictions(k,:)
%     line([a(1) b(1)], [a(2) b(2)], 'Color', 'cyan')
% end

xlim([-1 21])
ylim([-1 21])

legend([
    "Access Points"
    "Offline Training Points"
    "Online Training Points (1)"
    "Online Training Points (2)"
    "Offline Training Point"
    "Online Testing Points (1)"
    "Online Testing Points (2)"
%     "Predictions"
], 'Location', 'east');

hold off;

csvwrite('output/train_small.csv', [Tr1_signals([1,300],:) Tr1_actual([1,300],:)])
csvwrite('output/train_XY_1.csv', [Tr1_signals Tr1_actual])
csvwrite('output/train_XY_2.csv', [Tr2_signals Tr2_actual])
csvwrite('output/train_XY_3.csv', [Tr3_signals Tr3_actual])
csvwrite('output/test_XY_1.csv', [Te1_signals Te1_actual])
csvwrite('output/test_XY_2.csv', [Te2_signals Te2_actual])
csvwrite('output/test_XY_3.csv', [Te3_signals Te3_actual])

function signals = calculateSignals(AP, input, alpha)
    NUM_ACCESS_POINTS = length(AP);

    signals = zeros(length(input),NUM_ACCESS_POINTS);

    for i = 1:length(input)
        D = sqrt(sum(((AP - input(i,:)) .^ 2)'))';
        PL0 = -40;
        X_sigma = randn(size(D)) * 0.5;
        PL_D = PL0 - 10*alpha*log(D) + X_sigma;
        signals(i,1:NUM_ACCESS_POINTS) = PL_D';
    end
end

function [signal, actualLocationSmall, actualLocation] = makeNWithKSamples(n, k, alpha, AP)
    actualLocationSmall = zeros(n, 2);
    actualLocation = zeros(n * k, 2);
    output = zeros(n * k, 2);

    for p = 0:n
        selectedRandomLocation = rand(1, 2) * 20;
        actualLocationSmall(p + 1, :) = selectedRandomLocation;
        for tr = 1:k
            output(k * p + tr,:) = selectedRandomLocation + (randn(1, 2) * 0.1);
            actualLocation(k * p + tr, :) = selectedRandomLocation;
        end
    end
    
    signal = calculateSignals(AP, output, alpha);
end



