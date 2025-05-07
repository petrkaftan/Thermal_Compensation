from gpytorch.constraints import GreaterThan, LessThan
import torch
import gpytorch
from gpytorch.settings import cholesky_jitter
from sklearn.preprocessing import StandardScaler
import pandas as pd

class GPy_module(object):

    def __init__(self):

        self.lr = 0.1   
        self.max_iter = 400
        self.history_size = 15
        self.max_attempts = 5

        torch.set_default_dtype(torch.float64)
    
    def train_model(self,df_X_train,df_y_train):

        # Scale
        self.scaler_x = StandardScaler()                        
        self.scaler_y = StandardScaler()

        self.X_train = pd.DataFrame(self.scaler_x.fit_transform(df_X_train), columns=df_X_train.columns).values
        self.y_train = pd.DataFrame(self.scaler_y.fit_transform(df_y_train), columns=df_y_train.columns).values     

        # Convert to Torch
        self.X_train = torch.from_numpy(self.X_train)
        self.y_train = torch.flatten(torch.from_numpy(self.y_train))

        for attempt in range(self.max_attempts):

            try:

                # Set up the likelihood and model
                self.likelihood = gpytorch.likelihoods.GaussianLikelihood().double()

                self.model = GPModel(self.X_train, self.y_train, self.likelihood).double()
                self.model.likelihood.noise_covar.initialize(noise=1e-2) 

                # Set the model to training mode
                self.model.train()
                self.likelihood.train()

                # Use the LBFGS optimizer
                optimizer = torch.optim.LBFGS([
                    {'params': self.model.parameters()},
                ], lr=self.lr, max_iter=self.max_iter, history_size=self.history_size)

                # "Loss" function: the marginal log likelihood
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

                # Closure function for L-BFGS
                def closure():
                    optimizer.zero_grad()
                    output = self.model(self.X_train)
                    self.loss = -mll(output, self.y_train)
                    self.loss.backward()
                    if torch.isnan(self.loss):
                        raise ValueError("NaN detected in loss!")
                    return self.loss

                # Training loop with L-BFGS
                for i in range(1):  # L-BFGS requires only one iteration, as it performs multiple updates within each call
                    with cholesky_jitter(1e-1):  # Add jitter 
                        optimizer.step(closure)
                        print(f'Final loss: {self.loss.item()}')
                break

            except RuntimeError as e:
                print(f"[GPy_module] Attempt {attempt+1} failed due to: {e}")
                if attempt == self.max_attempts - 1:
                    raise RuntimeError("GP training failed after maximum retries") from e
                else:
                    print("[GPy_module] Retrying training with new model initialization...")

        # Set the model to evaluation mode
        self.model.eval()
        self.likelihood.eval()

    def predict(self,df_X_pred):

        X_pred = torch.from_numpy(self.scaler_x.transform(df_X_pred)).double()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            y_pred_torch = self.likelihood(self.model(X_pred))

        y_pred = y_pred_torch.mean.numpy()
        lower, upper = y_pred_torch.confidence_region() ## Get standard deviation

        # Extract relevant outputs (prediction and standard deviation)
        y_pred = y_pred_torch.mean.numpy()
        y_var  = y_pred_torch.variance.detach().numpy()

        # Scale back the outputs
        y_pred = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1))

        # Get y_std
        y_std = (y_var.reshape(-1, 1) ** 0.5) * self.scaler_y.scale_[0]

        return y_pred, y_std

# Define the GP Model
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1])
            #gpytorch.kernels.MaternKernel(ard_num_dims=train_x.shape[1],nu=1.5)
        )
        self.covar_module.base_kernel.initialize(lengthscale=1.0)
        self.covar_module.initialize(outputscale=1.0)

        self.covar_module.base_kernel.lengthscale.constraint = gpytorch.constraints.Interval(0.01, 10000)
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)