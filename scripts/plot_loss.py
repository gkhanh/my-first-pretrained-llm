import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

def plot_training_progress(log_file="training_log.csv"):
    if not os.path.exists(log_file):
        print(f"Error: {log_file} not found. Have you started training yet?")
        return

    try:
        # Read the CSV file
        df = pd.read_csv(log_file)
        
        # Check if empty
        if len(df) < 2:
            print("Not enough data to plot yet.")
            return

        # Setup the plot
        plt.figure(figsize=(12, 6))
        
        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(df['token_count'] / 1e6, df['loss'], label='Training Loss', color='blue', alpha=0.7)
        
        # Add a smoothed trend line (rolling average)
        if len(df) > 10:
            df['loss_smooth'] = df['loss'].rolling(window=10).mean()
            plt.plot(df['token_count'] / 1e6, df['loss_smooth'], label='Smoothed (MA-10)', color='red', linewidth=2)
            
        plt.title('Training Loss over Time')
        plt.xlabel('Tokens Processed (Millions)')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Plot Speed
        plt.subplot(1, 2, 2)
        plt.plot(df['token_count'] / 1e6, df['speed'], label='Speed (t/s)', color='green', alpha=0.7)
        plt.title('Training Speed')
        plt.xlabel('Tokens Processed (Millions)')
        plt.ylabel('Tokens / Second')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Save the plot
        output_file = "training_progress.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        print(f"Graph saved to {output_file}")
        
        # Show plot window if possible (optional, might fail in pure terminal/WSL without X11)
        # plt.show() 
        
    except Exception as e:
        print(f"An error occurred plotting the data: {e}")

if __name__ == "__main__":
    plot_training_progress()

