%% =========================================================
%  SELF-HEALING AI — LORENZ SYSTEM
%  =========================================================
%  Features:
%    1. Simulate clean Lorenz attractor (ground truth)
%    2. Inject noise + spikes into x, y, z
%    3. AI prediction via linear extrapolation
%    4. Correction  — threshold-based fault detection & fix
%    5. Smoothing   — Gaussian-weighted moving average
%    6. Fragility map in (sigma, rho) parameter space
%    7. Forensic report: before/after, component, value changed
%
%  No toolboxes required.
%% =========================================================
clc; clear; close all;

fprintf('============================================================\n');
fprintf('  SELF-HEALING AI  --  LORENZ SYSTEM\n');
fprintf('============================================================\n\n');

rng(42);

%% ============================================================
%  SECTION 1: LORENZ SIMULATION (ground truth)
%% ============================================================
sigma_lorenz = 10;
rho_lorenz   = 28;
beta_lorenz  = 8/3;

dt  = 0.01;
T   = 10;
t   = 0:dt:T;
N   = length(t);

X = zeros(N,3);
X(1,:) = [1, 1, 1];

for i = 1:N-1
    k1 = lorenz_f(X(i,:),              sigma_lorenz, rho_lorenz, beta_lorenz);
    k2 = lorenz_f(X(i,:)+0.5*dt*k1,    sigma_lorenz, rho_lorenz, beta_lorenz);
    k3 = lorenz_f(X(i,:)+0.5*dt*k2,    sigma_lorenz, rho_lorenz, beta_lorenz);
    k4 = lorenz_f(X(i,:)+    dt*k3,    sigma_lorenz, rho_lorenz, beta_lorenz);
    X(i+1,:) = X(i,:) + (dt/6)*(k1+2*k2+2*k3+k4);
end

fprintf('Lorenz simulation: %d steps, dt=%.3f, T=%.1f s\n', N, dt, T);

%% ============================================================
%  SECTION 2: INJECT NOISE + SPIKES
%% ============================================================
noise_std  = [2.5, 3.0, 2.0];
spike_prob = 0.03;
spike_amp  = 15;

X_noisy   = X + randn(N,3) .* noise_std;
spike_mask= rand(N,3) < spike_prob;
X_noisy   = X_noisy + spike_mask .* (spike_amp*(2*rand(N,3)-1));

comp_names = {'x (Prandtl dir)', 'y (Rayleigh dir)', 'z (vertical)'};
labels     = {'x','y','z'};
fprintf('Noise injected: sx=%.1f sy=%.1f sz=%.1f, spike_prob=%.0f%%\n',...
    noise_std(1),noise_std(2),noise_std(3),spike_prob*100);

%% ============================================================
%  SECTION 3: AI PREDICTION (linear extrapolation)
%% ============================================================
ph     = 3;   % prediction horizon (steps)
X_pred = X_noisy;
for i = ph+1:N-ph
    slope = X_noisy(i,:) - X_noisy(i-1,:);
    X_pred(i+1:i+ph,:) = X_noisy(i,:) + (1:ph)'.*slope;
end

%% ============================================================
%  SECTION 4: CORRECTION (threshold fault detection)
%% ============================================================
resid  = abs(X_noisy - X_pred);
thresh = 2.5 * std(resid);

fault_mask  = resid > thresh;
X_corrected = X_noisy;
X_corrected(fault_mask) = X_pred(fault_mask);

n_per = sum(fault_mask);
fprintf('Faults detected: x=%d  y=%d  z=%d  total=%d\n',...
    n_per(1),n_per(2),n_per(3),sum(fault_mask(:)));

%% ============================================================
%  SECTION 5: SMOOTHING (Gaussian moving average)
%% ============================================================
win    = 11;
sig_g  = 2.0;
hw     = floor(win/2);
gw_raw = exp(-(-hw:hw).^2/(2*sig_g^2));
gw     = gw_raw/sum(gw_raw);

X_smooth = X_corrected;
for c = 1:3
    s = X_corrected(:,c);
    for i = hw+1:N-hw
        X_smooth(i,c) = sum(gw .* s(i-hw:i+hw)');
    end
end

%% ============================================================
%  SECTION 6: ERROR ANALYSIS
%% ============================================================
err_noisy = sqrt(mean((X_noisy    - X).^2));
err_corr  = sqrt(mean((X_corrected- X).^2));
err_sm    = sqrt(mean((X_smooth   - X).^2));
maxerr_n  = max(abs(X_noisy  - X));
maxerr_s  = max(abs(X_smooth - X));

fprintf('\n--- RMSE vs ground truth ---\n');
fprintf('  %-4s  %-10s  %-12s  %-10s\n','Comp','Noisy','Corrected','Smoothed');
for c = 1:3
    imp = (err_noisy(c)-err_sm(c))/err_noisy(c)*100;
    fprintf('  %-4s  %-10.4f  %-12.4f  %-10.4f  (+%.1f%%)\n',...
        labels{c},err_noisy(c),err_corr(c),err_sm(c),imp);
end

[~,wc] = max(err_noisy);   % worst component

%% ============================================================
%  SECTION 7: FORENSIC — find worst spike
%% ============================================================
diff_wc = abs(X_noisy(:,wc) - X(:,wc));
[max_ev, max_idx] = max(diff_wc);
t_sp   = t(max_idx);
v_gt   = X(max_idx,wc);
v_n    = X_noisy(max_idx,wc);
v_c    = X_corrected(max_idx,wc);
v_s    = X_smooth(max_idx,wc);

fprintf('\n--- FORENSIC REPORT ---\n');
fprintf('  Component     : %s\n', comp_names{wc});
fprintf('  Time          : t = %.3f s  (step %d)\n', t_sp, max_idx);
fprintf('  Ground truth  : %+.6f\n', v_gt);
fprintf('  Corrupted     : %+.6f  (error = %+.6f)\n', v_n, v_n-v_gt);
fprintf('  After correct : %+.6f  (error = %+.6f)\n', v_c, v_c-v_gt);
fprintf('  After smooth  : %+.6f  (error = %+.6f)\n', v_s, v_s-v_gt);
fprintf('  Value changed : %.6f  -->  %.6f\n', v_n, v_c);
err_red = (1 - abs(v_s-v_gt)/max_ev)*100;
fprintf('  Error reduced : %.1f%%\n', err_red);

%% ============================================================
%  SECTION 8: FRAGILITY MAP (sigma x rho grid)
%% ============================================================
fprintf('\nBuilding fragility map...\n');
sg_range = linspace(5,20,25);
rh_range = linspace(15,45,25);
fmap     = zeros(length(rh_range),length(sg_range));

for si = 1:length(sg_range)
    for ri = 1:length(rh_range)
        sg = sg_range(si); rh = rh_range(ri);
        Nt = 200; Xf = zeros(Nt,3); Xf(1,:)=[1,1,1];
        for ii=1:Nt-1
            k1=lorenz_f(Xf(ii,:),sg,rh,beta_lorenz);
            k2=lorenz_f(Xf(ii,:)+0.5*dt*k1,sg,rh,beta_lorenz);
            k3=lorenz_f(Xf(ii,:)+0.5*dt*k2,sg,rh,beta_lorenz);
            k4=lorenz_f(Xf(ii,:)+dt*k3,sg,rh,beta_lorenz);
            Xf(ii+1,:)=Xf(ii,:)+(dt/6)*(k1+2*k2+2*k3+k4);
        end
        Xn=Xf+2.5*randn(Nt,3);
        Xp=Xn;
        for ii=3:Nt-1
            slp=Xn(ii,:)-Xn(ii-1,:); Xp(ii+1,:)=Xn(ii,:)+slp;
        end
        r2=abs(Xn-Xp); thr2=2.5*std(r2); fm=r2>thr2;
        Xc2=Xn; Xc2(fm)=Xp(fm);
        Xs2=Xc2;
        for ii=hw+1:Nt-hw
            Xs2(ii,:)=sum(gw.*Xc2(ii-hw:ii+hw,:));
        end
        fmap(ri,si)=mean(sqrt(mean((Xs2-Xf).^2,1)));
    end
end
fmap_n=(fmap-min(fmap(:)))/(max(fmap(:))-min(fmap(:)));

[~,ri_nom]=min(abs(rh_range-rho_lorenz));
[~,si_nom]=min(abs(sg_range-sigma_lorenz));
fprintf('Fragility at nominal (s=%.0f, r=%.0f): %.3f\n',...
    sigma_lorenz,rho_lorenz,fmap_n(ri_nom,si_nom));

%% ============================================================
%  SECTION 9: PLOTS
%% ============================================================
comp_cols = {[0.3 0.8 1],[0.5 1 0.4],[1 0.7 0.3]};

figure('Name','Self-Healing AI Lorenz','Color','k','Position',[20 20 1500 1000]);

% Row 1: component traces
for c=1:3
    subplot(4,3,c);
    plot(t,X(:,c),'Color',[0.5 0.5 0.5],'LineWidth',1.2); hold on;
    plot(t,X_noisy(:,c),'Color',[1 0.4 0.4],'LineWidth',0.7,'LineStyle',':');
    plot(t,X_smooth(:,c),'Color',comp_cols{c},'LineWidth',1.8);
    if c==wc
        xline(t_sp,'w--','LineWidth',1.5);
        text(t_sp+0.1,v_n,' spike','Color','w','FontSize',7);
    end
    set(gca,'Color','k','XColor','w','YColor','w');
    xlabel('t (s)','Color','w','FontSize',8);
    ylabel(labels{c},'Color','w','FontSize',9);
    title(sprintf('Component %s',labels{c}),'Color','w','FontWeight','bold');
    legend({'Original','Noisy','Healed'},'TextColor','w','Color','k',...
        'EdgeColor',[0.3 0.3 0.3],'FontSize',6,'Location','best');
    grid on; set(gca,'GridColor',[0.25 0.25 0.25]);
end

% Row 2: predicted vs actual
for c=1:3
    subplot(4,3,3+c);
    plot(t,X(:,c),'Color',[0.5 0.5 0.5],'LineWidth',1.2); hold on;
    plot(t,X_pred(:,c),'Color',[1 0.75 0.2],'LineWidth',1,'LineStyle','--');
    set(gca,'Color','k','XColor','w','YColor','w');
    xlabel('t (s)','Color','w','FontSize',8);
    ylabel(labels{c},'Color','w');
    title(sprintf('Prediction vs Actual  %s',labels{c}),'Color','w','FontWeight','bold');
    legend({'Actual','Predicted'},'TextColor','w','Color','k',...
        'EdgeColor',[0.3 0.3 0.3],'FontSize',6,'Location','best');
    grid on; set(gca,'GridColor',[0.25 0.25 0.25]);
end

% Row 3: RMSE bar + error trace + improvement %
subplot(4,3,7);
errs=[err_noisy;err_corr;err_sm];
b=bar(errs,0.7,'grouped');
for ci=1:3, b(ci).FaceColor=comp_cols{ci}; b(ci).EdgeColor='none'; end
set(gca,'Color','k','XColor','w','YColor','w',...
    'XTickLabel',{'Noisy','Corrected','Smoothed'},'XTickLabelRotation',12);
ylabel('RMSE','Color','w'); title('RMSE per stage','Color','w','FontWeight','bold');
legend({'x','y','z'},'TextColor','w','Color','k','EdgeColor',[0.3 0.3 0.3],'FontSize',6);
grid on; set(gca,'GridColor',[0.25 0.25 0.25]);

subplot(4,3,8);
ae_n=abs(X_noisy(:,wc)-X(:,wc));
ae_s=abs(X_smooth(:,wc)-X(:,wc));
plot(t,ae_n,'Color',[1 0.4 0.4],'LineWidth',1); hold on;
plot(t,ae_s,'Color',[0.3 0.9 0.5],'LineWidth',1.4);
set(gca,'Color','k','XColor','w','YColor','w');
xlabel('t (s)','Color','w'); ylabel('|error|','Color','w');
title(sprintf('Error trace — component %s',labels{wc}),'Color','w','FontWeight','bold');
legend({'Before healing','After healing'},'TextColor','w','Color','k',...
    'EdgeColor',[0.3 0.3 0.3],'FontSize',6);
grid on; set(gca,'GridColor',[0.25 0.25 0.25]);

subplot(4,3,9);
imp=(err_noisy-err_sm)./err_noisy*100;
b2=bar(imp,0.6,'FaceColor','flat','EdgeColor','none');
b2.CData=cell2mat(comp_cols');
set(gca,'Color','k','XColor','w','YColor','w','XTickLabel',{'x','y','z'});
ylabel('% improvement','Color','w');
title('RMSE improvement','Color','w','FontWeight','bold');
yline(0,'w--');
for c=1:3
    text(c,imp(c)+0.5,sprintf('%.1f%%',imp(c)),'Color','w',...
        'HAlign','center','FontSize',9,'FontWeight','bold');
end
grid on; set(gca,'GridColor',[0.25 0.25 0.25]);

% Row 4: Fragility map + 3D attractor + zoom on worst event
subplot(4,3,10);
imagesc(sg_range,rh_range,fmap_n);
colormap(gca,frag_cmap()); cb=colorbar('Color','w');
cb.Label.String='Fragility'; cb.Label.Color='w';
hold on;
plot(sigma_lorenz,rho_lorenz,'wp','MarkerSize',14,'MarkerFaceColor','w','LineWidth',2);
text(sigma_lorenz+0.3,rho_lorenz+0.6,'Nominal','Color','w','FontSize',7,'FontWeight','bold');
contour(sg_range,rh_range,fmap_n,[0.3 0.3],'c-','LineWidth',1.5);
contour(sg_range,rh_range,fmap_n,[0.7 0.7],'r-','LineWidth',1.5);
set(gca,'Color','k','XColor','w','YColor','w','YDir','normal');
xlabel('\sigma','Color','w'); ylabel('\rho','Color','w');
title('Fragility map (\sigma, \rho)','Color','w','FontWeight','bold');

subplot(4,3,11);
plot3(X(:,1),X(:,2),X(:,3),'Color',[0.5 0.5 0.5],'LineWidth',0.6); hold on;
plot3(X_smooth(:,1),X_smooth(:,2),X_smooth(:,3),'Color',[0.3 0.8 1],'LineWidth',0.8);
set(gca,'Color','k','XColor','w','YColor','w','ZColor','w');
xlabel('x','Color','w'); ylabel('y','Color','w'); zlabel('z','Color','w');
title('3D attractor: original vs healed','Color','w','FontWeight','bold');
legend({'Original','Healed'},'TextColor','w','Color','k','EdgeColor',[0.3 0.3 0.3],'FontSize',6);
grid on; set(gca,'GridColor',[0.2 0.2 0.2]); view(-30,25);

subplot(4,3,12);
z1=max(1,max_idx-50); z2=min(N,max_idx+50);
tz=t(z1:z2);
plot(tz,X(z1:z2,wc),'Color',[0.5 0.5 0.5],'LineWidth',2); hold on;
plot(tz,X_noisy(z1:z2,wc),'Color',[1 0.4 0.4],'LineWidth',1.2,'LineStyle',':');
plot(tz,X_corrected(z1:z2,wc),'Color',[1 0.8 0.2],'LineWidth',1.2,'LineStyle','--');
plot(tz,X_smooth(z1:z2,wc),'Color',[0.3 0.9 0.5],'LineWidth',2);
xline(t_sp,'w--','LineWidth',1.5);
plot(t_sp,v_n,'rv','MarkerSize',10,'MarkerFaceColor','r','LineWidth',2);
plot(t_sp,v_s,'g^','MarkerSize',10,'MarkerFaceColor','g','LineWidth',2);
set(gca,'Color','k','XColor','w','YColor','w');
xlabel('t (s)','Color','w'); ylabel(labels{wc},'Color','w');
title(sprintf('Zoom: worst event  t=%.2f  comp=%s',t_sp,labels{wc}),...
    'Color','w','FontWeight','bold');
legend({'Truth','Corrupted','Corrected','Healed'},...
    'TextColor','w','Color','k','EdgeColor',[0.3 0.3 0.3],'FontSize',6,'Location','best');
grid on; set(gca,'GridColor',[0.25 0.25 0.25]);

sgtitle('Self-Healing AI  --  Lorenz System  |  Prediction . Correction . Smoothing',...
    'Color','w','FontSize',12,'FontWeight','bold');

%% ============================================================
%  SECTION 10: CONSOLE SUMMARY
%% ============================================================
fprintf('\n============================================================\n');
fprintf('  FULL OUTPUT SUMMARY\n');
fprintf('============================================================\n');
fprintf('\nBEFORE CORRECTION:\n');
fprintf('  RMSE  x=%.4f  y=%.4f  z=%.4f\n',err_noisy(1),err_noisy(2),err_noisy(3));
fprintf('  MaxE  x=%.4f  y=%.4f  z=%.4f\n',maxerr_n(1),maxerr_n(2),maxerr_n(3));
fprintf('\nAFTER CORRECTION + SMOOTHING:\n');
fprintf('  RMSE  x=%.4f  y=%.4f  z=%.4f\n',err_sm(1),err_sm(2),err_sm(3));
fprintf('  MaxE  x=%.4f  y=%.4f  z=%.4f\n',maxerr_s(1),maxerr_s(2),maxerr_s(3));
fprintf('\nIMPROVEMENT:\n');
for c=1:3
    fprintf('  %s: %.4f --> %.4f  (%.1f%% better)\n',...
        labels{c},err_noisy(c),err_sm(c),(err_noisy(c)-err_sm(c))/err_noisy(c)*100);
end
fprintf('\nCOMPONENT WITH HIGHEST ERROR: %s\n',comp_names{wc});
fprintf('  Time            : t = %.3f s\n',t_sp);
fprintf('  Value BEFORE    : %.6f\n',v_n);
fprintf('  Value AFTER     : %.6f\n',v_s);
fprintf('  Ground truth    : %.6f\n',v_gt);
fprintf('  Changed FROM    : %.6f\n',v_n);
fprintf('  Changed TO      : %.6f  (correction step)\n',v_c);
fprintf('  Error reduced by: %.1f%%\n',err_red);
fprintf('\nFRAGILITY MAP:\n');
fnom=fmap_n(ri_nom,si_nom);
fprintf('  Nominal operating point (s=%.0f,r=%.0f) fragility=%.3f\n',...
    sigma_lorenz,rho_lorenz,fnom);
if fnom<0.3,   fprintf('  Zone: SAFE  (cyan contour = boundary)\n');
elseif fnom<0.7,fprintf('  Zone: WARNING\n');
else,           fprintf('  Zone: CRITICAL\n'); end
fprintf('\nDone.\n');

%% LOCAL FUNCTIONS
function dX = lorenz_f(X,sigma,rho,beta)
    dX = zeros(1,3);
    dX(1) = sigma*(X(2)-X(1));
    dX(2) = X(1)*(rho-X(3))-X(2);
    dX(3) = X(1)*X(2)-beta*X(3);
end

function cmap = frag_cmap()
    n=256;
    st=[0,.25,.5,.75,1];
    sc=[.05 .35 .15; .2 .7 .25; .9 .8 .1; .9 .35 .05; .55 .05 .2];
    cmap=zeros(n,3);
    for k=1:n
        v=(k-1)/(n-1);
        for s=2:5
            if v<=st(s)
                tt=(v-st(s-1))/(st(s)-st(s-1));
                cmap(k,:)=(1-tt)*sc(s-1,:)+tt*sc(s,:); break;
            end
        end
    end
end
