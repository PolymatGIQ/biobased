# Descriptor Analysis
# Version: 2025.11.18 (Light)

import os
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors
from scipy import stats as scipy_stats
import plotly.graph_objects as go

# ===== USER CONFIGURATION =====
USE_PRESELECTED_DESCRIPTORS = False  # Set to True to use your own descriptor list

# Pre-selected descriptors
PRESELECTED_DESCRIPTORS = []


# Get all available RDKit descriptors (AUTOCORR2D excluded)
def RDKit_descriptors():
    descriptors = {}
    keep_autocorr = {f"AUTOCORR2D_{i:02d}" for i in range(5, 30, 5)}

    for name, function in Descriptors.__dict__.items():
        if callable(function) and not name.startswith('_'):
            # Exclude FpDensityMorgan descriptors
            if name.startswith("FpDensityMorgan"):
                continue

            # Keep only specific AUTOCORR2D descriptors
            if not name.startswith("AUTOCORR2D") or name in keep_autocorr:
                descriptors[name] = function

    return descriptors


# Analyze descriptor values for a list of SMILES
def analyze_descriptors(smiles_list, descriptor_dict):
    """Calculate descriptor values and basic statistics."""
    results = {}
    molecules = [(i, Chem.MolFromSmiles(s)) for i, s in enumerate(smiles_list)]
    valid_mol = [(i, mol) for i, mol in molecules if mol is not None]

    if len(valid_mol) == 0:
        print("Warning: No valid molecules found in the SMILES list!")
        return results

    print("Calculating descriptors...")
    for desc_name, desc_func in tqdm(descriptor_dict.items(), desc="Progress"):
        values = []
        for i, mol in valid_mol:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    value = desc_func(mol)
                    values.append(float(value))
            except Exception:
                continue

        if values:
            try:
                values = np.array(values, dtype=float)
                mask = np.isfinite(values)
                finite_values = values[mask]

                if len(finite_values) > 0:
                    # Check for extreme outliers (beyond reasonable range)
                    min_val = float(np.min(finite_values))
                    max_val = float(np.max(finite_values))

                    # Flag if max is extremely large (possible calculation error)
                    if max_val > 1e30:
                        print(f"Warning: {desc_name} has extreme values (max={max_val:.2e}). Skipping.")
                        continue

                    results[desc_name] = {
                        'values': finite_values.tolist(),
                        'min': min_val,
                        'max': max_val,
                        'mean': float(np.mean(finite_values)),
                        'std': float(np.std(finite_values))
                    }
            except Exception as e:
                continue

    return results


# Calculate correlation between descriptors and target property
def calculate_descriptor_property_correlation(results, y_values):
    """Calculate Pearson correlation between each descriptor and the target property."""
    correlations = {}
    y_values = np.array(y_values)

    print("\nCalculating descriptor-property correlations...")
    for desc_name, stats in tqdm(results.items(), desc="Progress"):
        if 'values' not in stats or len(stats['values']) == 0:
            continue

        descriptor_values = np.array(stats['values'])

        # Make sure both arrays have the same length
        min_len = min(len(descriptor_values), len(y_values))
        desc_vals = descriptor_values[:min_len]
        y_vals = y_values[:min_len]

        # Remove any NaN or infinite values
        valid_mask = np.isfinite(desc_vals) & np.isfinite(y_vals)
        desc_vals_clean = desc_vals[valid_mask]
        y_vals_clean = y_vals[valid_mask]

        if len(desc_vals_clean) > 2:
            try:
                corr_coef, p_value = scipy_stats.pearsonr(desc_vals_clean, y_vals_clean)

                # Skip if correlation is NaN or infinite
                if not np.isfinite(corr_coef) or not np.isfinite(p_value):
                    continue

                correlations[desc_name] = {
                    'correlation': float(corr_coef),
                    'abs_correlation': float(abs(corr_coef)),
                    'p_value': float(p_value),
                    'n_samples': len(desc_vals_clean)
                }
            except Exception:
                continue

    return correlations


# Calculate correlations between descriptors
def calculate_descriptor_correlations(results, descriptor_list):
    """Calculate pairwise correlations between descriptors."""

    # Find minimum length across all descriptors
    min_length = float('inf')
    for desc in descriptor_list:
        if desc in results:
            values = results[desc].get('values', [])
            if len(values) > 0 and len(values) < min_length:
                min_length = len(values)

    if min_length == float('inf') or min_length == 0:
        return pd.DataFrame()

    # Build the data matrix
    descriptor_data = {}
    for desc in descriptor_list:
        if desc in results:
            values = results[desc].get('values', [])
            if len(values) >= min_length:
                descriptor_data[desc] = values[:min_length]

    # Create DataFrame and calculate correlation matrix
    df = pd.DataFrame(descriptor_data)
    corr_matrix = df.corr()

    return corr_matrix


# Remove redundant descriptors
def remove_redundant_descriptors(top_descriptors, property_correlations,
                                 descriptor_corr_matrix, threshold=0.9):
    """
    Remove redundant descriptors based on descriptor-descriptor correlations.
    Keep the one with the higher property correlation.
    """
    print(f"\nRemoving redundant descriptors (threshold={threshold})...")

    descriptors_to_remove = set()
    removal_log = []

    # Check all pairs
    for i in range(len(descriptor_corr_matrix.columns)):
        for j in range(i + 1, len(descriptor_corr_matrix.columns)):
            desc1 = descriptor_corr_matrix.columns[i]
            desc2 = descriptor_corr_matrix.columns[j]

            # Skip if either already marked for removal
            if desc1 in descriptors_to_remove or desc2 in descriptors_to_remove:
                continue

            # Check correlation between descriptors
            desc_corr = descriptor_corr_matrix.iloc[i, j]

            if abs(desc_corr) >= threshold:
                # Get property correlations
                prop_corr1 = property_correlations[desc1]['abs_correlation']
                prop_corr2 = property_correlations[desc2]['abs_correlation']

                # Remove the one with lower property correlation
                if prop_corr1 >= prop_corr2:
                    descriptors_to_remove.add(desc2)
                    removal_log.append({
                        'kept': desc1,
                        'removed': desc2,
                        'desc_corr': desc_corr,
                        'kept_prop_corr': prop_corr1,
                        'removed_prop_corr': prop_corr2
                    })
                else:
                    descriptors_to_remove.add(desc1)
                    removal_log.append({
                        'kept': desc2,
                        'removed': desc1,
                        'desc_corr': desc_corr,
                        'kept_prop_corr': prop_corr2,
                        'removed_prop_corr': prop_corr1
                    })

    # Create final list without redundant descriptors
    final_descriptors = [d for d in top_descriptors if d not in descriptors_to_remove]

    # Print removal summary
    if removal_log:
        print(f"\nRemoved {len(descriptors_to_remove)} redundant descriptors:")
        print(f"{'Kept':<30} {'Removed':<30} {'Desc Corr':>10} {'Kept Prop':>10} {'Removed Prop':>10}")
        print("-" * 100)
        for entry in removal_log:
            print(f"{entry['kept']:<30} {entry['removed']:<30} "
                  f"{entry['desc_corr']:>10.3f} "
                  f"{entry['kept_prop_corr']:>10.4f} "
                  f"{entry['removed_prop_corr']:>10.4f}")
    else:
        print("No redundant descriptors found with the given threshold.")

    return final_descriptors, removal_log


# Create heatmap with property + descriptors
def create_heatmap_with_property(results, property_correlations,
                                 final_descriptors, y_values,
                                 y_name, dataset_name):
    """
    Create a heatmap showing:
    - Row/Column 1: Target property
    - Remaining rows/cols: Top descriptors
    All correlations are calculated and displayed.
    """
    print("\nGenerating correlation heatmap...")

    # Prepare data for correlation calculation
    # Find minimum length
    min_length = len(y_values)
    for desc in final_descriptors:
        if desc in results:
            desc_len = len(results[desc]['values'])
            if desc_len < min_length:
                min_length = desc_len

    # Build data dictionary with property first
    data_dict = {y_name: y_values[:min_length]}

    for desc in final_descriptors:
        if desc in results:
            data_dict[desc] = results[desc]['values'][:min_length]

    # Create DataFrame and calculate full correlation matrix
    df_full = pd.DataFrame(data_dict)
    corr_matrix_full = df_full.corr()

    # Get labels (property first, then descriptors)
    labels = [y_name] + final_descriptors

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix_full.values,
        x=labels,
        y=labels,
        colorscale='RdYlGn',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix_full.values, 2),
        texttemplate='%{text}',
        textfont={"size": 12},
        hoverongaps=False,
        colorbar=dict(title="Correlation")
    ))

    fig.update_layout(
        title=f'Correlation Heatmap: {y_name} + Top {len(final_descriptors)} Descriptors',
        height=1200,
        width=1200,
        xaxis_tickangle=-45,
        xaxis_title='',
        yaxis_title='',
        font=dict(size=12)
    )

    # Save the plot
    output_dir = './Analyze'
    os.makedirs(output_dir, exist_ok=True)
    # Save as HTML
    html_filename = f'{output_dir}/{dataset_name}_Final_Heatmap.html'
    fig.write_html(html_filename)
    print(f"Heatmap saved to: {html_filename}")

    # Update fonts for PNG export
    fig.update_layout(font=dict(size=20), title_font=dict(size=20))
    fig.update_traces(textfont=dict(size=18))

    # Save as PNG
    png_filename = f'{output_dir}/{dataset_name}_Final_Heatmap.png'
    fig.write_image(png_filename, width=2000, height=2000, scale=2)
    print(f"Heatmap saved to: {png_filename}")

    return corr_matrix_full


# Main analysis function
def analyze_dataset():
    """Main function to run the streamlined analysis."""

    print("DESCRIPTOR ANALYSIS")
    print(f"==================")

    # Get filename from user
    while True:
        filename = input("\nPlease enter the name of your Excel file (without .xlsx): ")
        address = f"./Data/{filename}.xlsx"
        if not os.path.exists(address):
            print(f"Error: File '{address}' not found!")
            continue
        break

    try:
        # Read Excel file
        print(f"\nReading {filename}...")
        df = pd.read_excel(address)

        # Get column names
        while True:
            x_name = input("Please enter the name of your SMILES column: ")
            y_name = input("Please enter the name of your property column: ")
            x_match = [col for col in df.columns if col.upper() == x_name.upper()]
            y_match = [col for col in df.columns if col.upper() == y_name.upper()]

            if x_match and y_match:
                x_name, y_name = x_match[0], y_match[0]
                break
            print(f"Error: One or more columns not found in {filename}")
            continue

        print(f"\nProcessing SMILES from '{x_name}' column...")
        smiles_list = df[x_name].tolist()
        y_values = df[y_name].values

        print(f"Total molecules: {len(smiles_list)}")
        print(f"Property: {y_name}")
        print(f"Property range: [{np.min(y_values):.4f}, {np.max(y_values):.4f}]")

        # Get descriptors
        print("\nGetting RDKit descriptors...")
        descriptor_dict = RDKit_descriptors()
        print(f"Found {len(descriptor_dict)} RDKit descriptors")

        # Calculate descriptor values
        results = analyze_descriptors(smiles_list, descriptor_dict)
        print(f"Successfully calculated {len(results)} descriptors")

        # Calculate descriptor-property correlations
        property_correlations = calculate_descriptor_property_correlation(results, y_values)
        print()
        print(f"Calculated correlations for {len(property_correlations)} descriptors")

        # FILTER OUT NaN VALUES BEFORE SORTING
        valid_correlations = {
            desc: corr_data
            for desc, corr_data in property_correlations.items()
            if np.isfinite(corr_data['abs_correlation'])
        }

        print(f"Valid correlations (non-NaN): {len(valid_correlations)}")

        # Sort by absolute correlation
        sorted_correlations = sorted(
            valid_correlations.items(),
            key=lambda x: x[1]['abs_correlation'],
            reverse=True
        )

        # Display top correlations
        print("\n" + "=" * 80)
        print(f"TOP 20 DESCRIPTORS BY CORRELATION WITH {y_name}")
        print("=" * 80)
        print(f"{'Rank':<6} {'Descriptor':<35} {'Correlation':>12} {'|Corr|':>10} {'P-value':>12}")
        print("-" * 80)
        for i, (desc, corr_data) in enumerate(sorted_correlations[:20], 1):
            print(f"{i:<6} {desc:<35} {corr_data['correlation']:>+12.4f} {corr_data['abs_correlation']:>10.4f} {corr_data['p_value']:>12.2e}")

        # ===== CHECK IF USING PRE-SELECTED DESCRIPTORS =====
        if USE_PRESELECTED_DESCRIPTORS:
            print("\n" + "=" * 80)
            print("USING PRE-SELECTED DESCRIPTORS (Skipping redundancy removal)")
            print("=" * 80)

            # Validate that pre-selected descriptors exist in the results
            valid_preselected = []
            missing_descriptors = []

            for desc in PRESELECTED_DESCRIPTORS:
                if desc in valid_correlations:
                    valid_preselected.append(desc)
                else:
                    missing_descriptors.append(desc)

            if missing_descriptors:
                print(f"\nWarning: {len(missing_descriptors)} descriptors not found or have invalid correlations:")
                for desc in missing_descriptors:
                    print(f"  - {desc}")

            print(f"\nUsing {len(valid_preselected)} pre-selected descriptors")

            # Use pre-selected list as final descriptors
            final_descriptors = valid_preselected
            removal_log = []  # Empty removal log

            # Print the descriptors being used
            print("\n" + "=" * 80)
            print("PRE-SELECTED DESCRIPTORS (Ranked by correlation with property)")
            print("=" * 80)

            # Sort by correlation for display
            preselected_with_corr = [
                (desc, valid_correlations[desc]['abs_correlation'])
                for desc in valid_preselected
            ]
            preselected_with_corr.sort(key=lambda x: x[1], reverse=True)

            print(f"{'Rank':<6} {'Descriptor':<35} {'Correlation':>12} {'|Corr|':>10}")
            print("-" * 80)
            for i, (desc, abs_corr) in enumerate(preselected_with_corr, 1):
                corr_data = valid_correlations[desc]
                print(f"{i:<6} {desc:<35} {corr_data['correlation']:>+12.4f} {abs_corr:>10.4f}")

            # Set heatmap size to all pre-selected descriptors
            heatmap_n = len(final_descriptors)
            heatmap_descriptors = final_descriptors

        else:
            # ===== ORIGINAL WORKFLOW: Redundancy removal =====

            # Get user input for number of descriptors and redundancy threshold
            try:
                n_descriptors = int(input("\nHow many top descriptors to analyze? (default: 50): ") or "50")
                redundancy_threshold = float(input("Redundancy threshold (0.0-1.0, default: 0.9): ") or "0.9")
            except ValueError:
                print("Invalid input. Using defaults: 50 descriptors, threshold=0.9")
                n_descriptors = 50
                redundancy_threshold = 0.9

            # Get top N descriptors
            top_descriptors = [desc for desc, _ in sorted_correlations[:n_descriptors]]

            print(f"\nAnalyzing top {n_descriptors} descriptors for redundancy...")

            # Calculate descriptor-descriptor correlations
            descriptor_corr_matrix = calculate_descriptor_correlations(results, top_descriptors)

            if descriptor_corr_matrix.empty:
                print("Error: Could not calculate descriptor correlations.")
                return

            # Remove redundant descriptors
            final_descriptors, removal_log = remove_redundant_descriptors(
                top_descriptors,
                valid_correlations,
                descriptor_corr_matrix,
                threshold=redundancy_threshold
            )

            print(f"\nFinal descriptor count: {len(final_descriptors)}")
            print(f"Original: {n_descriptors} → Removed: {n_descriptors - len(final_descriptors)} → Final: {len(final_descriptors)}")

            # Ask user how many to include in heatmap
            try:
                heatmap_n = int(input(f"\nHow many descriptors to include in heatmap? (max: {len(final_descriptors)}, default: 28): ") or "28")
                heatmap_n = min(heatmap_n, len(final_descriptors))
            except ValueError:
                heatmap_n = min(28, len(final_descriptors))

            # Take top N for heatmap
            heatmap_descriptors = final_descriptors[:heatmap_n]

        print(f"\nCreating heatmap with {heatmap_n} descriptors + property...")

        # Create heatmap
        corr_matrix = create_heatmap_with_property(
            results,
            valid_correlations,
            heatmap_descriptors,
            y_values,
            y_name,
            filename
        )

        # Save final descriptor list to CSV
        output_dir = './Analyze'
        os.makedirs(output_dir, exist_ok=True)

        # Different filename based on mode
        if USE_PRESELECTED_DESCRIPTORS:
            csv_filename = f'{output_dir}/{filename}_PreSelected_Descriptors.csv'
            heatmap_note = "Pre-Selected"
        else:
            csv_filename = f'{output_dir}/{filename}_Optimized_Descriptors.csv'
            heatmap_note = "Optimized (Redundancy Removed)"

        df_results = pd.DataFrame([
            {
                'Rank': i,
                'Descriptor': desc,
                'Correlation': valid_correlations[desc]['correlation'],
                'Abs_Correlation': valid_correlations[desc]['abs_correlation'],
                'P_Value': valid_correlations[desc]['p_value']
            }
            for i, desc in enumerate(final_descriptors, 1)
        ])
        df_results.to_csv(csv_filename, index=False)
        print(f"Descriptor list saved to: {csv_filename}")

        # Save removal log only if redundancy removal was performed
        if removal_log:
            removal_filename = f'{output_dir}/{filename}_Removed_Redundant.csv'
            df_removed = pd.DataFrame(removal_log)
            df_removed.to_csv(removal_filename, index=False)
            print(f"Redundant descriptor pairs saved to: {removal_filename}")

        # Print summary
        print("\n" + "=" * 80)
        if USE_PRESELECTED_DESCRIPTORS:
            print(f"PRE-SELECTED DESCRIPTORS IN HEATMAP ({heatmap_n} descriptors)")
        else:
            print(f"DESCRIPTORS IN HEATMAP (Top {heatmap_n})")
        print("=" * 80)
        print(f"{'Rank':<6} {'Descriptor':<35} {'Correlation':>12} {'|Corr|':>10}")
        print("-" * 80)
        for i, desc in enumerate(heatmap_descriptors, 1):
            corr_data = valid_correlations[desc]
            print(f"{i:<6} {desc:<35} {corr_data['correlation']:>+12.4f} {corr_data['abs_correlation']:>10.4f}")

        print("\n" + "=" * 80)
        print("Analysis complete!")
        print("=" * 80)
        print(f"\nGenerated files:")
        if USE_PRESELECTED_DESCRIPTORS:
            print(f"  1. {filename}_Final_Heatmap.html - Interactive correlation heatmap (Pre-selected)")
            print(f"  2. {filename}_PreSelected_Descriptors.csv - Your pre-selected descriptor list with correlations")
        else:
            print(f"  1. {filename}_Final_Heatmap.html - Interactive correlation heatmap")
            print(f"  2. {filename}_Optimized_Descriptors.csv - Optimized descriptor list (redundancy removed)")
            if removal_log:
                print(f"  3. {filename}_Removed_Redundant.csv - Log of removed redundant pairs")

    except Exception as e:
        print(f"\nError processing {filename}: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    analyze_dataset()

