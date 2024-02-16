% Fatemeh Moghadasian - 401129050
% Optimization Methods_project FALL 2023-24
clc; clear all; close all;

% Parameters
rho = 1.05;             % Air density (kg/m^3)
s = 0.09;               % Wing area (m^2)
b = 0.914;              % Wing span (m)
c = 0.01;               % Middle chord line (m)
I_xx = 0.16;            % Roll angle moment of inertia
c_lp = -0.15;           % Dimensionless coefficient for roll angle
c_ldeltaa = 0.005;      % Dimensionless coefficient for ailerons movement
v = 16;                 % Velocity (m/s)

% Aerodynamic coefficients
L_p = ((rho * s * v * b^2) / (4 * I_xx)) * c_lp;
L_deltaa = ((rho * s * v^2 * b) / (2 * I_xx)) * c_ldeltaa;

% Simulation time
t_sim = 100;

% transfer function of system
G = tf([L_deltaa], [1, -L_p, 0]);
Gf = feedback(G, 1);
figure(1)
step(Gf, t_sim);
grid on;
hold on

% PSO parameters
itr = 200;
N =300;
c1 = 2;                 % Cognitive coefficient
c2 = 2;                 % Social coefficient
wmax = 0.9;
wmin = 0.4;
var = 3;
lb = 0;                 % Lower bound
ub = 10;               % Upper bound
c_cf = 0;

% Reference
r_s = 1;

% Initialization
for m = 1:N
    for n = 1:var
        v(m, n) = 0;    % Velocity particles
        x(m, n) = lb + rand * (ub - lb);    % Position particles
        xp(m, n) = x(m, n);
    end

    % Model parameters
    kp = x(m, 1);
    ki = x(m, 2);
    kd = x(m, 3);

    % Simulation model
    Gc = pid(kp, ki, kd);
    Gcf = feedback(Gc * G, 1);
    [y_step, t] = step(Gcf, t_sim);

    % Objective function - ITSE
    ffb(m) = sum((t .* (y_step - r_s)).^2); 
end

% Find the best value
[fg, Gbest_location] = min(ffb);
xg = x(Gbest_location, :);

% Initialize variables
max_overshoot = -inf;
settling_time = -1;
control_coefficients = zeros(itr, var);
SSE = zeros(1, itr);

% PSO optimization loop
for i = 1:itr
    w = wmax - ((wmax - wmin) * i / itr);
    for m = 1:N
        for n = 1:var
            v(m, n) = (w * v(m, n)) + (c1 * rand * (xp(m, n) - x(m, n))) + (c2 * rand * (xg(n) - x(m, n))); % Update velocity
            x(m, n) = x(m, n) + v(m, n); % Update position
        end

        % Check bounds
        x(m, :) = max(min(x(m, :), ub), lb);

        % Model parameters
        kp = x(m, 1);
        ki = x(m, 2);
        kd = x(m, 3);

        % Simulation model
        Gc = pid(kp, ki, kd);
        Gcf = feedback(Gc * G, 1);
        [y_step, t] = step(Gcf, t_sim); 

        % Objective function - ITSE
        ff(m) = sum((t .* (y_step - r_s)).^2); % Calculate ITSE

        % Compare local
        if ff(m) < ffb(m)
            ffb(m) = ff(m);
            xp(m, :) = x(m, :);
        end
    end

    % Update global best
    [Bfg, location] = min(ffb);
    if Bfg < fg
        fg = Bfg; % New global value
        xg = xp(location, :); % Position of var
    end

    c_cf = c_cf + 1;
    best_cf_pso(c_cf) = fg;
    t = 1:c_cf;
    figure(2)
    plot(t, best_cf_pso, "k", "LineWidth", 2)
    xlabel("Iteration")
    ylabel("ITSE")
    grid on;
    hold on

    control_coefficients(i, :) = xg;
    SSE(i) = sum((y_step - r_s).^2);
end

Min_ITSE = fg

% PID parameters
kp = xg(1)
ki = xg(2)
kd = xg(3)
SSE_Value = SSE(end)

% Simulation model
Gc = pid(kp, ki, kd);
Gcf = feedback(Gc * G, 1);
figure(1)
step(Gcf, t_sim);
legend("System", "PID_PSO");
grid on;
[y_step,t_sim]=step(Gcf, t_sim);

[step_info, ~] = stepinfo(y_step, t_sim);
settling_time = step_info.SettlingTime; 
max_overshoot = step_info.Overshoot;
fprintf('Maximum Overshoot: %.2f%%\n', max_overshoot);
fprintf('Settling Time: %.2f\n', settling_time);

figure(3);
plot(1:itr, control_coefficients(:, 1), 'r', 'LineWidth', 1.5);
hold on;
plot(1:itr, control_coefficients(:, 2), 'g', 'LineWidth', 1.5);
plot(1:itr, control_coefficients(:, 3), 'b', 'LineWidth', 1.5);
xlabel('Iteration');
ylabel('Control Coefficients');
legend('Kp', 'Ki', 'Kd');
grid on;