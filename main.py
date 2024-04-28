import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

class CellphonePricePredictionApp:
    def __init__(self, master):
        self.master = master
        self.master.title('Cellphone Price Prediction')

        # Initialize sliders list at the very beginning
        self.sliders = []

        # Load data
        self.data = pd.read_csv('CellphoneData.csv')
        self.X = self.data.drop('Price', axis=1)  # Store DataFrame directly for later use in creating widgets
        self.y = self.data['Price'].values

        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                random_state=42)

        # Initialize and fit the model
        self.model = XGBRegressor()
        self.model.fit(self.X_train, self.y_train)

        # Now that self.sliders is initialized, create widgets
        self.create_widgets()

    def create_widgets(self):
        # Iterate over the columns of self.X to create sliders
        for i, column in enumerate(self.X.columns):
            label = tk.Label(self.master, text=column + ': ')
            label.grid(row=i, column=0)
            current_val_label = tk.Label(self.master, text='0.0')
            current_val_label.grid(row=i, column=2)
            slider = ttk.Scale(self.master, from_=self.X[column].min(), to=self.X[column].max(), orient="horizontal",
                               command=lambda val, label=current_val_label: label.config(text=f'{float(val):.2f}'))
            slider.grid(row=i, column=1)
            self.sliders.append((slider, current_val_label))

        # Button to trigger price prediction
        predict_button = tk.Button(self.master, text='Predict Price', command=self.predict_price)
        predict_button.grid(row=len(self.X.columns), columnspan=3)

    def predict_price(self):
        inputs = [float(slider.get()) for slider, _ in self.sliders]
        price = self.model.predict([inputs])
        messagebox.showinfo('Predicted Price', f'The predicted cellphone price is ${price[0]:.2f}')


if __name__ == '__main__':
    root = tk.Tk()
    app = CellphonePricePredictionApp(root)
    root.mainloop()


