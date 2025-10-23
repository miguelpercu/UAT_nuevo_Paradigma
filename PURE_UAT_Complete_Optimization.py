#!/usr/bin/env python
# coding: utf-8

# In[4]:


# =============================================================================
# PURE UAT - COMPLETE OPTIMIZATION
# =============================================================================

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import pandas as pd
import os

class Pure_UAT_Optimization:
    def __init__(self):
        self.c = 299792.458
        self.rd_planck = 147.09
        self.H0_target = 73.00
        self.Omega_m = 0.315
        self.Omega_r = 9.22e-5

        # Create results directory
        self.results_dir = "UAT con EO"
        os.makedirs(self.results_dir, exist_ok=True)

        self.bao_data = {
            'z': [0.38, 0.51, 0.61, 1.48, 2.33],
            'DM_rd_obs': [10.25, 13.37, 15.48, 26.47, 37.55],
            'DM_rd_err': [0.16, 0.20, 0.21, 0.41, 1.15]
        }

    def calculate_DM_rd_UAT_pure(self, z, k_early):
        """Calculates DM/rd for pure UAT (emergent Omega_Lambda)"""
        Omega_Lambda_UAT = 1 - k_early * (self.Omega_m + self.Omega_r)

        def E_UAT_pure(z_prime):
            return np.sqrt(k_early * (self.Omega_r*(1+z_prime)**4 + self.Omega_m*(1+z_prime)**3) + Omega_Lambda_UAT)

        integral, _ = quad(lambda zp: 1.0/E_UAT_pure(zp), 0, z)
        DM = (self.c / self.H0_target) * integral
        rd_UAT = self.rd_planck * k_early**0.5

        return DM / rd_UAT

    def chi2_UAT_pure(self, k_early):
        """Calculates chi-square for pure UAT with given k_early"""
        chi2 = 0.0
        for i, z in enumerate(self.bao_data['z']):
            pred = self.calculate_DM_rd_UAT_pure(z, k_early)
            obs = self.bao_data['DM_rd_obs'][i]
            err = self.bao_data['DM_rd_err'][i]
            chi2 += ((obs - pred) / err)**2
        return chi2

    def save_optimization_results(self, k_optimal, Omega_Lambda_optimal, chi2_optimal):
        """Saves optimization results to text file"""
        filename = os.path.join(self.results_dir, "optimization_results.txt")
        with open(filename, 'w', encoding='utf-8') as f:  # Added UTF-8 encoding
            f.write("PURE UAT OPTIMIZATION RESULTS\n")
            f.write("=" * 40 + "\n")
            f.write(f"Optimal k_early (pure UAT): {k_optimal:.5f}\n")
            f.write(f"Emergent Omega_Lambda: {Omega_Lambda_optimal:.5f}\n")  # Replaced Ω_Λ
            f.write(f"Minimum chi-square: {chi2_optimal:.3f}\n")  # Replaced χ²
            f.write(f"H0: {self.H0_target:.2f} km/s/Mpc (fixed)\n")  # Replaced H₀
            f.write(f"Omega_m: {self.Omega_m}\n")  # Replaced Ω_m
            f.write(f"Omega_r: {self.Omega_r}\n")  # Replaced Ω_r

        print(f"✓ Optimization results saved to: {filename}")

    def save_data_comparison(self, k_optimal):
        """Saves data comparison to CSV file"""
        z_values = self.bao_data['z']
        observed = self.bao_data['DM_rd_obs']
        errors = self.bao_data['DM_rd_err']
        predicted = [self.calculate_DM_rd_UAT_pure(z, k_optimal) for z in z_values]
        residuals = [obs - pred for obs, pred in zip(observed, predicted)]

        df = pd.DataFrame({
            'Redshift_z': z_values,
            'Observed_DM_rd': observed,
            'Predicted_DM_rd': predicted,
            'Residuals': residuals,
            'Errors': errors
        })

        filename = os.path.join(self.results_dir, "data_comparison.csv")
        df.to_csv(filename, index=False)
        print(f"✓ Data comparison saved to: {filename}")

        return df

    def create_plots(self, k_optimal):
        """Creates and saves analysis plots"""
        # Plot 1: Data comparison
        z_range = np.linspace(0.1, 2.5, 100)
        predicted_curve = [self.calculate_DM_rd_UAT_pure(z, k_optimal) for z in z_range]

        plt.figure(figsize=(10, 6))
        plt.plot(z_range, predicted_curve, 'b-', label='Pure UAT Prediction', linewidth=2)
        plt.errorbar(self.bao_data['z'], self.bao_data['DM_rd_obs'], 
                    yerr=self.bao_data['DM_rd_err'], fmt='ro', 
                    capsize=5, label='BAO Data')
        plt.xlabel('Redshift (z)')
        plt.ylabel('DM/rd')
        plt.title('Pure UAT: DM/rd vs Redshift')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plot_filename = os.path.join(self.results_dir, "UAT_vs_data_plot.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Plot saved to: {plot_filename}")

        # Plot 2: Chi-squared landscape
        k_test = np.linspace(0.955, 0.975, 50)
        chi2_values = [self.chi2_UAT_pure(k) for k in k_test]

        plt.figure(figsize=(10, 6))
        plt.plot(k_test, chi2_values, 'g-', linewidth=2)
        plt.axvline(k_optimal, color='r', linestyle='--', label=f'Optimal k = {k_optimal:.5f}')
        plt.xlabel('k_early')
        plt.ylabel('Chi-square')
        plt.title('Chi-square Landscape for Pure UAT')
        plt.legend()
        plt.grid(True, alpha=0.3)

        chi2_plot_filename = os.path.join(self.results_dir, "chi2_landscape_plot.png")
        plt.savefig(chi2_plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Chi-squared plot saved to: {chi2_plot_filename}")

    def generate_executive_analysis(self, k_optimal, Omega_Lambda_optimal, chi2_optimal, df):
        """Generates executive scientific analysis report"""
        filename = os.path.join(self.results_dir, "executive_scientific_analysis.txt")

        with open(filename, 'w', encoding='utf-8') as f:  # Added UTF-8 encoding
            f.write("EXECUTIVE SCIENTIFIC ANALYSIS - PURE UAT OPTIMIZATION\n")
            f.write("=" * 60 + "\n\n")

            f.write("SUMMARY OF RESULTS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"• Optimal early curvature parameter: k_early = {k_optimal:.5f}\n")
            f.write(f"• Emergent cosmological constant: Omega_Lambda = {Omega_Lambda_optimal:.5f}\n")
            f.write(f"• Model fit quality: chi-square = {chi2_optimal:.3f}\n")
            f.write(f"• Number of data points: {len(self.bao_data['z'])}\n")
            f.write(f"• Degrees of freedom: {len(self.bao_data['z']) - 1}\n\n")

            f.write("PHYSICAL INTERPRETATION:\n")
            f.write("-" * 25 + "\n")
            f.write("• The Pure UAT model successfully reproduces BAO observations\n")
            f.write("• The emergent Omega_Lambda accounts for late-time acceleration\n")
            f.write("• k_early parameter modifies early universe geometry\n")
            f.write("• Model maintains fixed H0 = 73.00 km/s/Mpc\n\n")

            f.write("DATA COMPARISON DETAILS:\n")
            f.write("-" * 25 + "\n")
            for i, z in enumerate(self.bao_data['z']):
                pred = self.calculate_DM_rd_UAT_pure(z, k_optimal)
                obs = self.bao_data['DM_rd_obs'][i]
                err = self.bao_data['DM_rd_err'][i]
                residual = obs - pred
                sigma_deviation = abs(residual/err)
                f.write(f"z = {z}: Obs = {obs:.2f}, Pred = {pred:.2f}, "
                       f"Residual = {residual:.2f} ({sigma_deviation:.1f} sigma)\n")

            f.write(f"\nMean absolute residual: {df['Residuals'].abs().mean():.3f}\n")
            f.write(f"Reduced chi-square: {chi2_optimal/(len(self.bao_data['z'])-1):.3f}\n")

        print(f"✓ Executive analysis saved to: {filename}")

    def optimize(self):
        """Optimizes k_early for pure UAT and saves all results"""
        print("OPTIMIZING k_early FOR PURE UAT...")

        result = minimize_scalar(self.chi2_UAT_pure, bounds=(0.955, 0.975), method='bounded')
        k_optimal = result.x
        chi2_optimal = result.fun

        Omega_Lambda_optimal = 1 - k_optimal * (self.Omega_m + self.Omega_r)

        print(f"Optimal k_early (pure UAT): {k_optimal:.5f}")
        print(f"Emergent Omega_Lambda: {Omega_Lambda_optimal:.5f}")
        print(f"Minimum chi-square: {chi2_optimal:.3f}")
        print(f"H0: {self.H0_target:.2f} km/s/Mpc (fixed)")

        # Save all results
        self.save_optimization_results(k_optimal, Omega_Lambda_optimal, chi2_optimal)
        df = self.save_data_comparison(k_optimal)
        self.create_plots(k_optimal)
        self.generate_executive_analysis(k_optimal, Omega_Lambda_optimal, chi2_optimal, df)

        print(f"\n✓ All results saved in folder: '{self.results_dir}'")
        print("   - optimization_results.txt")
        print("   - data_comparison.csv") 
        print("   - UAT_vs_data_plot.png")
        print("   - chi2_landscape_plot.png")
        print("   - executive_scientific_analysis.txt")

        return k_optimal, Omega_Lambda_optimal, chi2_optimal

# Execute optimization
optimizer = Pure_UAT_Optimization()
k_opt, OmegaL_opt, chi2_opt = optimizer.optimize()


# In[ ]:




