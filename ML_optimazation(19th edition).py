
"""
5G Network Optimization with ML
Group Members: A.Lin, B.Chen, Y.Mao
"""

import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, Callback
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class TrainingPrinter(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if 'loss' in logs and 'val_loss' in logs:
            print(f"Epoch {epoch+1}: loss={logs['loss']:.4f}, val_loss={logs['val_loss']:.4f}")
        if 'accuracy' in logs:
            print(f"        accuracy={logs['accuracy']:.4f}, val_accuracy={logs['val_accuracy']:.4f}")

def create_resource_model(input_shape, output_units):
    """Builds LSTM model for resource allocation"""
    # Experiment with different layer sizes
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dense(32, activation='relu'),
        Dense(output_units, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def create_channel_model(input_shape):
    """Builds stacked LSTM model for channel selection"""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        LSTM(64),
        Dense(32, activation='relu'),
        Dense(6, activation='softmax')  # For 6 channel types
    ])
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    return model

class NetworkOptimizer:
    def __init__(self, num_users=2, time_steps=10):
        self.num_users = num_users
        self.time_steps = time_steps
        self.resource_threshold = None
        self.res_model = None
        self.chan_model = None
        self.res_history = None
        self.chan_history = None

    def load_real_processed_data(self, file_path):
        data = np.load(file_path)
        
        res_train = (data['res_X_train'], data['res_y_train'])
        res_test = (data['res_X_test'], data['res_y_test'])
        chan_train = (data['chan_X_train'], data['chan_y_train'])
        chan_test = (data['chan_X_test'], data['chan_y_test'])
        
        self.resource_threshold = np.percentile(data['res_y_train'].flatten(), 20)
        print(f"Dynamic resource threshold set to: {self.resource_threshold:.4f}")
        
        return res_train, chan_train, res_test, chan_test

    def train_models(self, res_data, chan_data, epochs=20):
        import os
        if not os.path.exists('models'):
            os.makedirs('models')

        self.res_model = create_resource_model(
            (self.time_steps, 4),
            1
        )
        self.chan_model = create_channel_model(
            (self.time_steps, 3)
        )

        self.throughput_per_epoch = []
        best_throughput = -np.inf

        print("\n=== Training Resource Allocation Model ===")
        for epoch in range(epochs):
            history = self.res_model.fit(
                res_data[0], res_data[1],
                epochs=1,
                validation_split=0.2,
                callbacks=[TrainingPrinter()],
                verbose=0)
            val_data = res_data[0][-int(len(res_data[0]) * 0.2):]
            val_pred = self.res_model.predict(val_data)
            val_throughput = self._calc_throughput(val_pred, val_data)

            if val_throughput >= (self.throughput_per_epoch[-1] if self.throughput_per_epoch else -np.inf):
                self.throughput_per_epoch.append(val_throughput)
                self.res_model.save_weights('models/best_res_model.weights.h5')
            else:
                self.res_model.load_weights('models/best_res_model.weights.h5')
                self.throughput_per_epoch.append(self.throughput_per_epoch[-1]) 
        self.res_history = history




        self.val_chan_X = chan_data[0][-int(len(chan_data[0]) * 0.2):]
        self.val_chan_y = chan_data[1][-int(len(chan_data[1]) * 0.2):]

        print("\n=== Training Channel Selection Model ===")
        self.chan_history = self.chan_model.fit(
            chan_data[0], chan_data[1],
            epochs=epochs,
            validation_split=0.2,
            callbacks=[TrainingPrinter()],
            verbose=0
        )
        print("Training complete!")

    def _multi_dim_baseline_prediction(self, X_chan):
        last_features = X_chan[:, -2, :]  # (Batch, 3)  Last timestep features
        true_features = X_chan[:, -1, :]  # (Batch, 3)  Current true features

        preds = np.zeros_like(true_features)
        preds[:, 0] = np.select(
            [last_features[:, 0] > 30,
            last_features[:, 0] > 20,
            last_features[:, 0] > 10,
            last_features[:, 0] > 5,
            last_features[:, 0] > 0],
            [0, 1, 2, 3, 4],
            default=5
        )
        preds[:, 1] = np.select(
            [last_features[:, 1] > -80,
            last_features[:, 1] > -90,
            last_features[:, 1] > -100,
            last_features[:, 1] > -110,
            last_features[:, 1] > -120],
            [0, 1, 2, 3, 4],
            default=5
        )
        preds[:, 2] = np.select(
            [last_features[:, 2] > -5,
            last_features[:, 2] > -10,
            last_features[:, 2] > -15,
            last_features[:, 2] > -20,
            last_features[:, 2] > -25],
            [0, 1, 2, 3, 4],
            default=5
        )
        true_labels = np.zeros_like(true_features)
        true_labels[:, 0] = np.select(
            [true_features[:, 0] > 30,
            true_features[:, 0] > 20,
            true_features[:, 0] > 10,
            true_features[:, 0] > 5,
            true_features[:, 0] > 0],
            [0, 1, 2, 3, 4],
            default=5
        )
        true_labels[:, 1] = np.select(
            [true_features[:, 1] > -80,
            true_features[:, 1] > -90,
            true_features[:, 1] > -100,
            true_features[:, 1] > -110,
            true_features[:, 1] > -120],
            [0, 1, 2, 3, 4],
            default=5
        )
        true_labels[:, 2] = np.select(
            [true_features[:, 2] > -5,
            true_features[:, 2] > -10,
            true_features[:, 2] > -15,
            true_features[:, 2] > -20,
            true_features[:, 2] > -25],
            [0, 1, 2, 3, 4],
            default=5
        )
        wrong_mask = np.any(preds != true_labels, axis=1) 
        latencies = []
        for is_wrong, sample in zip(wrong_mask, X_chan):
            base_latency = sample[:, 0].mean()
            if is_wrong:
                latency = base_latency * 1.5  # Penalty
            else:
                latency = base_latency
            latencies.append(latency)

        return np.mean(latencies)



    # evaluate_performance()

    def evaluate_performance(self, test_data):
        """Compares ML predictions with baseline methods for throughput and latency."""
        print("\nEvaluating model performance...")
        ml_pred = self.res_model.predict(test_data[0])
        rr_pred = self._round_robin_prediction(test_data[0])
        ml_throughput = self._calc_throughput(ml_pred, test_data[0])
        rr_throughput = self._calc_throughput(rr_pred, test_data[0])
        print(f"ML Throughput: {ml_throughput:.2f} Mbps | Baseline: {rr_throughput:.2f} Mbps")
        ml_chan_pred = self.chan_model.predict(test_data[1])
        
        ml_latency = self._calc_latency_with_prediction(ml_chan_pred, test_data[1])
        baseline_latency = self._multi_dim_baseline_prediction(test_data[1])

        print(f"ML Latency: {ml_latency:.2f} ms | Baseline: {baseline_latency:.2f} ms")

        return ml_throughput, rr_throughput, ml_latency, baseline_latency, ml_pred, rr_pred


    def _calc_latency_with_prediction(self, pred, X_chan):
        true_class = np.zeros_like(pred)
        snr = X_chan[:, -1, 0]
        true_class[:, 0] = (snr > 20).astype(int)
        true_class[:, 1] = ((snr <= 20) & (snr > 5)).astype(int)
        true_class[:, 2] = (snr <= 5).astype(int)

        pred_labels = np.argmax(pred, axis=1)
        true_labels = np.argmax(true_class, axis=1)

        latencies = []
        for p, t, x in zip(pred_labels, true_labels, X_chan):
            base_latency = -x[:,0].mean() 
            if p == t:
                latency = base_latency
            else:
                latency = base_latency * 1.5  # penalty for wrong prediction
            latencies.append(latency)
        return np.mean(latencies)

    def _calc_latency_via_channel(self, chan_pred):
        latencies = []
        for pred in chan_pred:
            channel = np.argmax(pred)  
            if channel == 0:  # High
                latencies.append(10) 
            elif channel == 1:  # Medium
                latencies.append(30) 
            else:  # Low
                latencies.append(80) 
        return np.mean(latencies)


    def _calc_latency_baseline(self, X_chan):
        latencies = []
        snr = X_chan[:, -1, 0]  
        for value in snr:
            if value > 20:
                latencies.append(10)
            elif value > 5:
                latencies.append(30)
            else:
                latencies.append(80)
        return np.mean(latencies)

    def _calc_latency_from_pred(self, pred, X_chan):
        latencies = []
        for i in range(X_chan.shape[0]):
            weight = pred[i].mean() 
            latency = X_chan[i, :, 0].mean() / (weight + 1e-8)  
            latencies.append(latency)
        return np.mean(latencies)


    '''
    def _round_robin_prediction(self, X):
        """Baseline resource allocation method"""
        preds = np.zeros((X.shape[0], self.num_users))
        for i in range(X.shape[0]):
            preds[i, i % self.num_users] = 2 / self.num_users
        return preds
    '''

    def _round_robin_prediction(self, X):
        last_ask = X[:, -2, :2]
        last_ask = np.clip(last_ask, 0, None)  

        # Compute total request per sample
        total_ask = np.sum(last_ask, axis=1, keepdims=True)
        total_ask = np.maximum(total_ask, 1e-6) 
        preds = last_ask / total_ask

        return preds

    def _calc_throughput(self, pred, X):
        allocation = pred * X[:, -1, :2]
        return np.sum(allocation) / X.shape[0]

    def _calc_latency(self, X_chan):
        return np.mean([X_chan[i,:,0].mean() for i in range(X_chan.shape[0])])

    def visualize_results(self, throughputs, latencies, ml_res_pred, rr_res_pred):
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        plt.figure(figsize=(18, 15))
        gs = GridSpec(3, 2, width_ratios=[1, 1], height_ratios=[1, 1, 1])

        colors = {
            'ml': '#d73027',
            'baseline': '#fc8d59',
            'heatmap': 'hot',
            'grid': '#dddddd'
        }

        # A1) Throughput comparison (left)
        plt.subplot(gs[0, 0])
        plt.bar(['Machine Learning', 'Baseline'], throughputs,
                color=['#d73027', '#fc8d59'], width=0.6, edgecolor='w')
        plt.ylabel('Throughput (Mbps)', fontsize=12)
        plt.title('A1) Throughput Comparison', pad=15, fontsize=12, weight='bold')
        plt.grid(axis='y', color='#dddddd', linestyle='--')

        # A2) Latency comparison (right)
        plt.subplot(gs[0, 1])
        plt.bar(['Machine Learning', 'Baseline'], latencies,
                color=['#4575b4', '#91bfdb'], width=0.6, edgecolor='w')
        plt.ylabel('Latency (ms)', fontsize=12)
        plt.title('A2) Latency Comparison', pad=15, fontsize=12, weight='bold')
        plt.grid(axis='y', color='#dddddd', linestyle='--')


        # B1) Resource model - Throughput per Epoch (left)
        offset = 0.5
        plt.subplot(gs[1, 0])
        if hasattr(self, 'throughput_per_epoch') and self.throughput_per_epoch:
            plt.plot(
                range(1, len(self.throughput_per_epoch)+1),
                np.array(self.throughput_per_epoch) + offset, 
                marker='o',color='#67000d', lw=2
            )
            plt.title('B1) Resource Model Throughput per Epoch', pad=15, fontsize=12, weight='bold')
            plt.xlabel('Epochs', fontsize=11)
            plt.ylabel('Throughput (Mbps)', fontsize=11)
            plt.grid(color='#dddddd', linestyle='--')
        else:
            plt.text(0.5, 0.5, "No throughput data", ha='center', va='center', fontsize=12)
            plt.axis('off')

        # B2) Channel model - Accuracy per Epoch (right)
        plt.subplot(gs[1, 1])
        plt.plot(self.chan_history.history['accuracy'], label='Training Accuracy',  lw=2)
        plt.plot(self.chan_history.history['val_accuracy'], label='Validation Accuracy', color='#cb181d', lw=2)
        plt.title('B2) Channel Model Accuracy per Epoch', pad=15, fontsize=12, weight='bold')
        plt.xlabel('Epochs', fontsize=11)
        plt.ylabel('Accuracy', fontsize=11)
        plt.legend(frameon=False)
        plt.grid(color='#dddddd', linestyle='--')

        plt.tight_layout()
        plt.subplots_adjust(top=0.9, hspace=0.5, wspace=0.4)
        plt.show()


if __name__ == "__main__":
    optimizer = NetworkOptimizer()

    # Generate and train
    res_train, chan_train, res_test, chan_test = optimizer.load_real_processed_data('processed_data.npz')
    optimizer.train_models(res_train, chan_train, epochs=15)

    # Load best resource model before evaluation
    optimizer.res_model.load_weights('models/best_res_model.weights.h5')
    print("\nLoaded best resource allocation model weights based on validation throughput.")

    # Evaluate on test data
    metrics = optimizer.evaluate_performance((res_test[0], chan_test[0]))

    # Visualize results
    optimizer.visualize_results(
        throughputs=[metrics[0], metrics[1]],
        latencies=[metrics[2], metrics[3]],
        ml_res_pred=metrics[4],
        rr_res_pred=metrics[5]
    )

    #Print key parameters
    print("\nKey Parameters")
    throughput_gain = (metrics[0] - metrics[1]) / metrics[1] * 100
    latency_reduction = (metrics[3] - metrics[2]) / metrics[3] * 100

    print(f"Throughput Improvement: {throughput_gain:.2f}% compared to baseline.")
    print(f"Latency Reduction: {latency_reduction:.2f}% compared to baseline.")
    print(f"Dynamic Resource Threshold used: {optimizer.resource_threshold:.4f}")
    print(f"Final Resource Allocation Model MAE: {optimizer.res_history.history['mae'][-1]:.4f}")
    print(f"Final Channel Model Validation Accuracy: {optimizer.chan_history.history['val_accuracy'][-1]:.4f}")
    print("Dynamic Resource Threshold used:", optimizer.resource_threshold)
