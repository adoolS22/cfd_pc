import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
from loguru import logger

def generate_signal_chart(symbol: str, df: pd.DataFrame, signal, output_path: str = '/tmp/chart.png'):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_plot = df.copy()
        
        # Ensure proper datetime index for mplfinance
        if 'timestamp' in df_plot.columns:
            df_plot.index = pd.to_datetime(df_plot['timestamp'], unit='ms')
        elif 'time' in df_plot.columns:
            df_plot.index = pd.to_datetime(df_plot['time'], unit='s')
        
        df_plot = df_plot.iloc[-80:] # last 80 candles
        
        # Add risk levels if they exist
        addplot_params = []
        if hasattr(signal, 'risk_levels') and getattr(signal, 'risk_levels', None):
            rl = signal.risk_levels
            length = len(df_plot)
            
            if hasattr(rl, 'entry') and rl.entry:
                addplot_params.append(mpf.make_addplot([float(rl.entry)]*length, color='blue', linestyle='--', width=2.0))
            if hasattr(rl, 'stop_loss') and rl.stop_loss:
                addplot_params.append(mpf.make_addplot([float(rl.stop_loss)]*length, color='red', linestyle='-.', width=2.0))
            
            if hasattr(rl, 'take_profit_1') and rl.take_profit_1:
                addplot_params.append(mpf.make_addplot([float(rl.take_profit_1)]*length, color='green', linestyle='-', width=1.5))
            if hasattr(rl, 'take_profit_2') and rl.take_profit_2:
                addplot_params.append(mpf.make_addplot([float(rl.take_profit_2)]*length, color='darkgreen', linestyle='-', width=1.5))
                
        style = mpf.make_mpf_style(base_mpf_style='charles', y_on_right=True)
        title = f"{symbol} - {signal.side} Signal"
        
        # Generate and save chart
        mpf.plot(df_plot, type='candle', style=style, title=title, volume=False, addplot=addplot_params, savefig=output_path, tight_layout=True)
        return output_path
    except Exception as e:
        logger.error(f"Failed to generate chart for {symbol}: {e}")
        return None
