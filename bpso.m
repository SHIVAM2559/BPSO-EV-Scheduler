% bpso_ev_v2v.m
% Binary PSO for EV charging/discharging scheduling (V2V-style)
% 5 EV example. Slot = 1 hour, 24 slots.
clear; close all; clc;
rng(1); % reproducible

%% ---------------- Problem setup ----------------
n_evs = 5;
n_slots = 24;
slot_dur = 1; % hours per slot
grid_capacity = 40; % kW grid max (hard soft-penalty)
price = ones(1,n_slots) * 1.0;
price(7:9) = 2.5;    % morning peak (7-9)
price(17:19) = 3.5;  % evening peak (17-19)
price(0+1:5) = 0.5;  % night cheap (1-5 index)

% EV parameters (you can edit)
% max_rate_kW, required_slots (integer), capacity_kWh, initial_soc (0-1), desired_soc
max_rate_kW = [7; 7; 3.3; 7; 3.3];                % column vector (n_evs x 1)
required_slots = [4; 3; 2; 5; 2];                 % number of full slots each EV needs
capacity_kWh = [40; 40; 20; 60; 24];              % battery capacity
initial_soc = [0.4; 0.6; 0.2; 0.5; 0.3];          % fraction (0..1)
desired_soc = [0.8; 0.9; 0.6; 0.9; 0.6];          % target SOC (for departure, or general)
arrival = [0; 6; 8; 0; 10];                       % slot indices (1-based) we'll use 0-23 adjust below
departure = [24; 18; 20; 24; 23];                 % slot end (exclusive)
% convert (MATLAB uses 1..24 indexing) -> we keep arrival as 1..24
arrival = max(1, arrival+1); departure = min(n_slots, departure);

% Charge/discharge efficiency
eta_c = 0.95;
eta_d = 0.95;

% Baseline (ASAP) schedule - charge as soon as arrived until requirement met
base_sched = zeros(n_evs, n_slots);
for i=1:n_evs
    need = required_slots(i);
    for t=arrival(i):departure(i)
        if need<=0, break; end
        base_sched(i,t) = 1;
        need = need - 1;
    end
end

% Helper to compute baseline metrics
[base_cost, base_peak, base_metrics] = evaluate_schedule(base_sched, zeros(n_evs,n_slots));
fprintf('Baseline cost: %.2f (arb units)   baseline peak: %.2f kW\n', base_cost, base_peak);

%% ---------------- BPSO parameters ----------------
n_particles = 80;
max_iter = 350;
w = 0.7; c1 = 1.5; c2 = 1.5;
vmax = 6; vmin = -6;

% Fitness weights (tune these)
w_cost = 1.0;         % weight for cost
w_peak = 0.5;         % weight for peak
penalty_unmet = 1e5;  % penalty weight for unmet required energy
penalty_grid = 1e4;   % penalty for grid capacity exceed
penalty_simul = 1e5;  % penalty for simultaneous charge+discharge

% Availability mask (1 means slot allowed for charge/discharge)
avail_mask = zeros(n_evs, n_slots);
for i=1:n_evs
    avail_mask(i, arrival(i):departure(i)) = 1;
end
% Flattening helpers
dim = 2 * n_evs * n_slots; % first half = charge bits, second half = discharge bits
mat_to_vec = @(C,D) [C(:); D(:)];
vec_to_mat = @(vec) deal(reshape(vec(1:n_evs*n_slots), [n_evs,n_slots]), ...
                         reshape(vec(n_evs*n_slots+1:end), [n_evs,n_slots]));

% Initialize particles (respecting availability)
particles = rand(n_particles, dim) < 0.05; % sparse start (few 1s)
v = randn(n_particles, dim);
% mask out impossible positions initially (no charge/discharge outside window)
mask_vec = [avail_mask(:); avail_mask(:)];
particles = particles .* mask_vec;

% initialize pbest and gbest
pbest = particles;
pbest_scores = inf(n_particles,1);
gbest = zeros(1,dim);
gbest_score = inf;

% Evaluate initial particles
for p=1:n_particles
    [C,D] = vec_to_mat(round(particles(p,:)));
    [fit, cost, unmet, outside, over, per_slot_power] = fitness_eval(C,D);
    pbest_scores(p) = fit;
    pbest(p,:) = particles(p,:);
    if fit < gbest_score
        gbest_score = fit;
        gbest = particles(p,:);
    end
end

% PSO main loop
best_history = zeros(max_iter,1);
for it=1:max_iter
    r1 = rand(n_particles, dim);
    r2 = rand(n_particles, dim);
    v = w*v + c1*r1.*(pbest - particles) + c2*r2.*(repmat(gbest, n_particles,1) - particles);
    % clamp velocity
    v(v>vmax) = vmax; v(v<vmin) = vmin;
    % sigmoid to get probability
    s = 1 ./ (1 + exp(-v));
    new_particles = rand(n_particles, dim) < s;
    % enforce availability mask
    new_particles = new_particles .* mask_vec;
    particles = new_particles;
    % Evaluate and update
    for p=1:n_particles
        [C,D] = vec_to_mat(round(particles(p,:)));
        [fit,cost,unmet,outside,over,per_slot_power] = fitness_eval(C,D);
        if fit < pbest_scores(p)
            pbest_scores(p) = fit;
            pbest(p,:) = particles(p,:);
        end
        if fit < gbest_score
            gbest_score = fit;
            gbest = particles(p,:);
        end
    end
    best_history(it) = gbest_score;
    if mod(it,50)==0
        fprintf('Iter %d: best fitness = %.3f\n', it, gbest_score);
    end
end

% Retrieve best solution
[Cbest,Dbest] = vec_to_mat(round(gbest));
[best_fit, best_cost, best_unmet, best_outside, best_over, best_per_slot_power] = fitness_eval(Cbest,Dbest);
best_peak = max(best_per_slot_power);

fprintf('\n--- Result ---\n');
fprintf('Baseline cost: %.2f    Baseline peak: %.2f kW\n', base_cost, base_peak);
fprintf('Optimized cost: %.2f    Optimized peak: %.2f kW\n', best_cost, best_peak);
fprintf('Cost reduction: %.2f%%  Peak reduction: %.2f%%\n', 100*(base_cost-best_cost)/base_cost, 100*(base_peak-best_peak)/base_peak);
fprintf('Unmet slots total: baseline %d, optimized %d\n', base_metrics.unmet, best_unmet);
fprintf('Grid violations (sum kW over capacity): %.2f kW (penalty counted)\n', best_over);

% Plot load profiles
slots = 1:n_slots;
base_load = (base_sched .* repmat(max_rate_kW,1,n_slots)) * ones(n_evs,1) ./ ones(n_evs,1); % per-slot aggregate
base_load = sum(base_sched .* repmat(max_rate_kW,1,n_slots),1);
opt_load = best_per_slot_power;

figure;
subplot(2,1,1);
plot(slots, base_load, '-o','DisplayName','Baseline load'); hold on;
plot(slots, opt_load, '-s','DisplayName','Optimized load');
plot(slots, price * (grid_capacity / max(price)) * 0.6, '--','DisplayName','Scaled price');
xlabel('Hour slot'); ylabel('Power (kW)'); title('Aggregate load');
legend; grid on;

subplot(2,1,2);
imagesc((Cbest)); title('Optimized charge schedule (EV x Slot)'); colorbar; xlabel('Slot'); ylabel('EV index');

%% ---------------- Local functions ----------------

function [fit, cost, unmet, outside, over, per_slot_power] = fitness_eval(C,D)
    % Nested evaluation function using workspace variables
    % We'll compute cost and penalties and return aggregated fitness
    % Access variables from outer scope via evalin (or embed everything as nested)
    % To simplify, we use globals via closure — but to keep clarity, we'll use caller workspace:
    params = evalin('base', 'struct(''n_evs'', n_evs, ''n_slots'', n_slots, ''slot_dur'', slot_dur, ...\n' ...
                      '''max_rate_kW'', max_rate_kW, ''eta_c'', eta_c, ''eta_d'', eta_d, ''capacity_kWh'', capacity_kWh, ...\n' ...
                      '''initial_soc'', initial_soc, ''required_slots'', required_slots, ''avail_mask'', avail_mask, ...\n' ...
                      '''grid_capacity'', grid_capacity, ''price'', price, ''penalty_unmet'', penalty_unmet, ''penalty_grid'', penalty_grid, ''penalty_simul'', penalty_simul, ''w_cost'', w_cost, ''w_peak'', w_peak);');
    n_evs = params.n_evs; n_slots = params.n_slots; slot_dur = params.slot_dur;
    max_rate_kW = params.max_rate_kW; eta_c = params.eta_c; eta_d = params.eta_d;
    capacity_kWh = params.capacity_kWh; initial_soc = params.initial_soc;
    required_slots = params.required_slots; avail_mask = params.avail_mask;
    grid_capacity = params.grid_capacity; price = params.price;
    penalty_unmet = params.penalty_unmet; penalty_grid = params.penalty_grid; penalty_simul = params.penalty_simul;
    w_cost = params.w_cost; w_peak = params.w_peak;

    % Force binary and availability
    C = double(C>0.5); D = double(D>0.5);
    C = C .* avail_mask; D = D .* avail_mask;
    % penalty for simultaneous charge+discharge
    simul = sum(sum((C & D)));
    simul_pen = simul * penalty_simul;

    % For each EV compute SOC evolution and detect unmet energy and invalid discharge attempts
    soc = initial_soc;
    soc_vec = zeros(n_evs, n_slots);
    unmet_slots = zeros(n_evs,1);
    invalid_discharge_cnt = 0;
    energy_from_grid = zeros(1, n_slots);
    for t=1:n_slots
        % per EV
        for i=1:n_evs
            % charge energy added this slot (kWh)
            e_ch = C(i,t) * max_rate_kW(i) * slot_dur * eta_c;
            e_dis = D(i,t) * max_rate_kW(i) * slot_dur / eta_d; % energy withdrawn from battery (kWh)
            % apply limits: cannot discharge if SOC * capacity < e_dis
            if e_dis > soc(i) * capacity_kWh(i)
                % invalid discharge -> penalize and set actual discharge to 0
                invalid_discharge_cnt = invalid_discharge_cnt + 1;
                e_dis = 0;
                D(i,t) = 0;
            end
            soc(i) = soc(i) + e_ch/capacity_kWh(i) - e_dis/capacity_kWh(i);
            if soc(i) < 0
                % should not happen if we checked, but clamp and penalize
                soc(i) = 0;
            elseif soc(i) > 1
                soc(i) = 1;
            end
            soc_vec(i,t) = soc(i);
            % grid energy: charging consumes from grid, discharging supplies to grid (negative)
            energy_from_grid(t) = energy_from_grid(t) + C(i,t)*max_rate_kW(i)*slot_dur - D(i,t)*max_rate_kW(i)*slot_dur;
        end
    end
    % compute per-slot power in kW (positive = draw from grid)
    per_slot_power = energy_from_grid / slot_dur; % since slot_dur in hours

    % cost: energy drawn from grid * price (if negative export, assume zero cost or no credit)
    draw_energy = max(energy_from_grid, 0);
    cost = sum(draw_energy .* price);

    % unmet: if EV didn't get required_slots of charging
    for i=1:n_evs
        actual_slots = sum(C(i,:));
        unmet_slots(i) = max(0, required_slots(i) - actual_slots);
    end
    unmet = sum(unmet_slots);

    % outside charging (should be zero because mask enforced) -- leftover
    outside = sum(sum((C + D) .* (~avail_mask)));

    % grid capacity violation (sum positive per-slot exceed)
    over_sum = sum(max(0, per_slot_power - grid_capacity));
    over = over_sum;

    % combine fitness
    fit = w_cost * cost + w_peak * max(per_slot_power) + penalty_unmet * unmet + penalty_grid * over + simul_pen + invalid_discharge_cnt * 1e4;

end

function [fit, cost, unmet, outside, over, per_slot_power] = evaluate_schedule(C, D)
    % convenience wrapper for initial baseline evaluation where D is zero
    [fit, cost, unmet, outside, over, per_slot_power] = fitness_eval(C,D);
end
