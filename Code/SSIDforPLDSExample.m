clear all
close all

%%%%%%%%%%%%%%%%%%%%%%%% generate artifical system and sample from it %%%%%%%%%%%%%%%%%%%%%%%%

Trials     = 100;
T          = 1000;	% length of each time-series, i.e. trial
xDim       = 15;     	% dimensionality of latent input
yDim       = 25;    	% dimensionality of observable
Bdim       = 0;     	% dimensionality of stimulus innovations
algo       = 'SVD'; 	% available algorithms 'SVD','CCA','N4SID'

[seq, trueparams] = GenerateArtificialPLDSdata(xDim,yDim,Trials,T,Bdim);
% load('ArtificialPLDSdata_5.mat')
% load('ArtificialPLDSdata_10.mat')
% load('ArtificialPLDSdata_15.mat')
% save ArtificialPLDSdata_15.mat seq trueparams

%%%%%%%%%%%%%%%%%%%%%%%%%% estimate parameters from data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[params,SIGBig] = FitPLDSParamsSSID(seq,xDim,'algo',algo);


%%%%%%%%%%%%%%%%%%%%%%%%%% some analysis %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('-----------------')
disp('Some analysis:')
fprintf('\nSubspace angle between true and estimated parameters: %d \n\n',subspace(trueparams.C,params.C))

% true eigen values
disp('True eigenspectrum:')
tru_eig = eig(trueparams.A);
sort(eig(trueparams.A))

% estimated eigen values
disp('Estimated spectrum')
sort(eig(params.A))
ident_eig = eig(params.A);

% plot the covariance matrix
covError = (sum(sum(((trueparams.C*trueparams.Q0*trueparams.C')-((params.C*params.Q0*params.C'))).^2)))/yDim^2
disp('Plot of true and estimated data covariance matrix')
figure()
subplot(2,1,1);
imagesc([trueparams.C*trueparams.Q0*trueparams.C'])
title(sprintf('True versus Estimated Covariance Matrix  \n %.0f Latent states \n (RMSE = %.4f)',xDim, covError))
ylabel('True')
subplot(2,1,2);
imagesc([params.C*params.Q0*params.C'])
ylabel('Estimated')
colorbar


% plot the identified eigenvalues
Error = (sum((real(tru_eig) - real(ident_eig)).^2 + (imag(tru_eig) - imag(ident_eig)).^2))/xDim;
figure()
     plot(real(tru_eig),imag(tru_eig),'ro') %   Plot real and imaginary parts
     hold on 
     plot(real(ident_eig),imag(ident_eig),'b*') %   Plot real and imaginary parts
     
     xlabel('Real')
     ylabel('Imaginary')
     legend ('True Eigenvalue','Identified Eigenvalue')
     title(sprintf('True versus Estimated Eigenvalues  \n %.0f Latent states \n (RMSE = %.4f)',xDim, Error))
     
     % initialize spike count
spike_count = zeros(1,T);
     % spike count
trial_data = seq.y;
for i_spike = 1:T
    spike_count(i_spike) = nnz(trial_data(:,i_spike));
end 

% plot trial 1 spike count
figure ()
plot(spike_count,'b*');
xlabel('time(10 ms temporal resolution)')
ylabel('spike count')
title('spike count data')

% plot histogramm
figure ()
histogram(spike_count)
xlabel('time(100 ms resolution)')
ylabel('spike count')
title('spike count distribution')

%% compare smoothed histogram

% smooth hist neuron 1 
y = seq.y
neuron_1= y(:,1);
[f,xi] = ksdensity(neuron_1);
figure()
plot(xi,f)



% convert estimated params into seq
cholQ  = chol(params.Q);
cholQ0 = chol(params.Q0); 

for tr=1:Trials
  seq_est(tr).x = zeros(xDim,T);
  seq_est(tr).x(:,1) = cholQ0'*randn(xDim,1);
  for t=2:T
      seq_est(tr).x(:,t) = params.A*seq_est(tr).x(:,t-1)+cholQ'*randn(xDim,1);
  end
 
  seq_est(tr).yr = params.C*seq_est(tr).x+repmat(params.d,1,T);
  
end

hold on
y_est = seq_est.yr
neuron_1_est= y_est(:,1);
[f_est,xi_est] = ksdensity(neuron_1_est);
plot(xi_est,f_est)



