import os
import warnings

# Silenciar TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Silenciar warnings generales
warnings.filterwarnings("ignore")



#os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # silencia avís symlinks en Windows
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all, 1 = info, 2 = warning, 3 = error


from tkinter import *
from tkinter import ttk
from tkinter import scrolledtext as st
from tkinter import messagebox as mb
import tkinter as tk
from tkcalendar import DateEntry
from datetime import datetime, timedelta
from datetime import date
from dateutil.relativedelta import relativedelta
import pytz
import json
from deep_translator import GoogleTranslator
import textwrap
import webbrowser

import yfinance as yf
import mplfinance as mpf
import plotly.graph_objects as go
#import ta  # Librería de análisis técnico
import pandas as pd
import pandas_ta as ta
import numpy as np
from numpy import nan as npNaN
import math


import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.ticker as mattk
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties
from matplotlib.widgets import Cursor
import matplotlib.dates as mdates # Necesitas importar este módulo
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor # ¡Nueva importación!

from sklearn.ensemble import RandomForestRegressor

from ratios import Financials, Balance, Quarterly_financials

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import time

from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, TimeDistributed, Flatten

from columnas import Tooltip, BtnToolTip
from functools import partial


################################
################################

class TkinterCallback(Callback):
    
    def __init__(self, app_instance):
        super().__init__()
        self.app = app_instance
        self.epoch_start_time = 0
        self.epoch_num = 0

    def on_train_begin(self, logs=None):
        pass
        #print('Empieza a entrenar')
        # after es una funcion que se ejecuta en el hilo principal 
        # Es una forma de programar una acción en la interfaz gráfica que se encuentra en el hilo principal
        # Actualiza el componente gráfico sin que se congele la pantalla
        #self.app.after(0, lambda: self.app.status_label.config(text="Entrenamiento en curso..."))
        
    def on_epoch_begin(self, epoch, logs=None):
        #print('Entra en epoch begin')
        self.epoch_num += 1
        # obtenemos la hora actual del sistema en milisegundos
        self.epoch_start_time = tk.Tcl().getint(tk.Tcl().eval('clock milliseconds'))
        
    
    def on_epoch_end(self, epoch, logs=None):
        pass
        #print('Entra en epoch end')
        # Usar .after() para ejecutar la actualización en el hilo principal de Tkinter
        #self.app.after(0, self.update_epoch_ui, epoch, logs)
    
    def on_train_end(self, logs=None):
        pass
        #print('Entra en train end')
        #self.app.after(0, lambda: self.app.status_label.config(text="Entrenamiento completado!"))
        
        
    def on_train_batch_begin(self,batch, logs):
        #self.app.after(0, self.app.actualizar_barra, batch, self.epoch_num)
        #print('Entra en train_batch_begin')
        self.app.actualizar_barra(batch, self.epoch_num)
        
    def on_train_batch_end(self, batch, logs):
        pass
        ##print('Entra en train batch end')  
        
    def on_test_begin(self, logs):
        pass
        #print('Entra en test begin')  
        #self.app.after(0, self.app.lecturas_entrenamiento)          
        
    def on_test_end(self, logs):
        pass
        #print('Entra en test end')  
        
        
    def on_predict_end(self, logs):
        pass
        #print('Entra en predict end')  
        
        
        
    def update_epoch_ui(self, epoch, logs):
        print('Pasa por el final de epoch dentro del evento ')
        pass

class Aplicacion(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Cotizaciones")
        self.datos = None
        
        
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width // 2) - (1300 // 2)
        y = (screen_height // 2) - (700 // 2) -25
        self.geometry(f"{1300}x{700}+{x}+{y}") 
        
        # Inidice vix
        self.vix = None
        
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self.cerrarVentana)
        
        # --- Configuración de expansión de la ventana raíz ---
        # Asegura que las filas y columnas de la ventana se expandan
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1) # La fila donde main_frame va a expandirse

        style = ttk.Style()
        style.theme_use('clam')
        
        # Creamos los frames
        # Parte superior fija (por ejemplo, barra de controles)
        self.top_frame = tk.Frame(self, bg="white", height=50)
        self.top_frame.pack(side="top", fill="x")
        self.top_frame.pack_propagate(False)  # ← Esto fija la altura

        # Frame que simula el borde inferior
        borde_inferior = tk.Frame(self.top_frame, bg="black", height=2)
        borde_inferior.pack(side="bottom", fill="x")

        # Parte inferior adaptable
        self.main_frame = tk.Frame(self, bg="white")
        self.main_frame.pack(side="top", expand=True, fill="both")

        # Creamos el combo
        # Lista de tickers del IBEX 35 (formato Yahoo Finance)
        '''
                
        tickers_ibex = [
            'NVDA','MSFT','AAPL','GOOG','AMZN','META','2222.SR','AVGO','TSLA','BRK.A','TSM','JPM',
            'WMT','LLY','XOM','JNJ','V','005930.KQ','NESN.SW','ROG.SW','PG','MA','MC.PA','CVX',
            'HD','ABBV','0700.HK',
            'ASML','KO','ORCL','PEP','NOVN.SW','ADBE','COST','PFE','MRK','TM','CRM','MCD',
            'INTC','AZN','IBM','SHEL','NFLX','SIE.DE','ULVR.L','DHR','QCOM','AMGN','TXN',
            'BHP','RIO','BABA','CVS','BKNG','LMT','GE','AIR.PA','AXP','INTU','GSK','SONY',
            'HMC','INGA.AS','SAN.MC','BBVA.MC','ENEL.MI','TTE','GLEN.L','HEIA.AS','ABI.BR',
            'BAYN.DE','MMM','SBUX','UBER','MRNA','PYPL','ZM','SQ','SPOT','RACE','ADYEN.AS',
            'ZAL.DE','MELI','SE','BIDU','JD','TEF.MC','IBE.MC','REP.MC','CABK.MC','ITX.MC',
            'GRF.MC','ENR.DE','LHA.DE','AENA.MC','NTGY.MC',                  
            'SAN.MC', 'BBVA.MC', 'IBE.MC', 'ITX.MC', 'TEF.MC', 'REP.MC', 'CABK.MC',
            'FER.MC', 'AMS.MC', 'CLNX.MC', 'GRF.MC', 'ENG.MC', 'NTGY.MC', 'ANA.MC',
            'ACS.MC', 'AENA.MC', 'MAP.MC', 'MEL.MC', 'SAB.MC', 'CIE.MC', 'COL.MC',
            'ELE.MC', 'LOG.MC', 'ROVI.MC', 'SOL.MC', 'ALM.MC', 'BKT.MC', 'CRX.MC',
            'ENC.MC', 'UNI.MC', 'SGRE.MC', 'PHM.MC', 'NXG.MC', 'EDR.MC', 'FCC.MC'

        ]
        '''
        # Leemos el CSV
        self.titulos = pd.read_csv("acciones.csv")
        self.titulos = self.titulos.sort_values(by='Ticker')
        
        # Extraer la columna como lista (sin duplicados)
        self.tickers = self.titulos["Ticker"].dropna().unique().tolist()
        self.tickers_f = self.titulos["Ticker"].dropna().unique().tolist()
        
        # Crear estilo personalizado
        style = ttk.Style()
        style.configure("Custom.TCombobox", font=("Arial", 12))

        Label(self.top_frame, text="Título:", bg='white', font=("Arial", 12)).place(x=10, y=10)
        self.ticker = tk.StringVar()
        self.combo_ticker = ttk.Combobox(self.top_frame, values=self.tickers, width=20, 
                    textvariable=self.ticker, style="Custom.TCombobox")
        self.ticker.set("Seleccione...")    
        self.combo_ticker.bind("<<ComboboxSelected>>", self.cambia_ticker)
        self.combo_ticker.place(x=60, y=10) 
        Label(self.top_frame, text="Opción:", bg='white', font=("Arial", 12)).place(x=235, y=10)
        options = ["Cotización", "Previsión", "Osciladores", "Datos", 'Información', 
                   'Análisis', 'Incongruencias', 'Recomendaciones', "Noticias"]
        
        
        self.option_graph = tk.StringVar()
        self.combo_graph = ttk.Combobox(self.top_frame, values=options, width=20, 
                    textvariable=self.option_graph, style="Custom.TCombobox")
        self.option_graph.set("Cotización")    
        self.combo_graph.bind("<<ComboboxSelected>>", self.cambia_graph)
        self.combo_graph.place(x=300, y=10) 

        Label(self.top_frame, text="Desde:", bg='white', font=("Arial", 12)).place(x=550, y=10)
        self.cal_inicio = DateEntry(self.top_frame)
        self.cal_inicio.place(x=610, y=10)
        self.cal_inicio.set_date(date(2023, 1, 1))
        

        Label(self.top_frame, text="Hasta:", bg='white', font=("Arial", 12)).place(x=720, y=10)
        self.cal_fin = cal_inicio = DateEntry(self.top_frame)
        self.cal_fin.place(x=775, y=10)   
        
        fecha_final = datetime.today()
        # Fecha inicial = fecha final - 10 años
        fecha_inicial = fecha_final - relativedelta(years=10)
        self.cal_inicio.set_date(fecha_inicial) 
        self.cal_fin.set_date(fecha_final) 
        
        
        
        self.btn_buscar = Button(self.top_frame, 
                                text="Buscar", 
                                font=("Arial", 12),
                                command=lambda: self.monta_cotizacion(self.ticker.get())
                            )

        self.btn_buscar.place(x=890, y=10)  
        
        
    
        
        self.mainloop() 
        
    def crear_y_posicionar(self, widget_class, master, place_args={}, **kwargs):
        widget = widget_class(master, **kwargs)
        widget.place(**place_args)
        return widget           
        
        
                
    def cambia_ticker(self, event):
           self.monta_cotizacion(event.widget.get())
           
    def dame_nombre(self, ticker):
        nRes = ""
        for index, row in self.titulos.iterrows():
            if row["Ticker"] == ticker:
                nRes = row["Nombre"]
        return nRes        
   
               
        
    def monta_cotizacion(self, ticker):
        # Obtener fecha actual como string con formato ISO
        fecha_ini = self.cal_inicio.get_date()
        fecha_fin = self.cal_fin.get_date()
        
        #fecha_actual = datetime.today().strftime('%Y-%m-%d') 
        start = fecha_ini
        end = fecha_fin
        if self.vix is None:
            vix_df = yf.download("^VIX", start=start, end=end)
            vix_limpio = vix_df.reset_index()
            # Aplanar columnas combinando los niveles del MultiIndex
            vix_limpio.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in vix_limpio.columns]
            vix_limpio = vix_limpio[['Date', 'Close_^VIX']]
            vix_limpio = vix_limpio.rename(columns={'Close_^VIX': 'VIX'})
            
            # 3. Solo necesitamos la Fecha y el Precio del VIX. La columna 'Ticker' es redundante.
            self.vix = vix_limpio[['Date', 'VIX']]
            # Convertir el campo Date en indice
            self.vix.set_index('Date', inplace=True)
            
        self.datos = yf.download(ticker, start=start, end=end, auto_adjust=True)

        self.info = yf.Ticker(ticker).info
        self.financials = yf.Ticker(ticker).financials
        self.balance_sheet = yf.Ticker(ticker).balance_sheet
        self.cash_flow = yf.Ticker(ticker).cash_flow
        self.recomendaciones = yf.Ticker(ticker).recommendations
        self.quarterly_financials = yf.Ticker(ticker).quarterly_financials
        self.news = yf.Ticker(ticker).news
        self.history = yf.Ticker(ticker).history(period="1d")
        #print('Noticias:', self.news)

        self.objetivo_LSTM = 0
        
        # Primero, verificar si las columnas son un MultiIndex
        # yfinance a veces devuelve un MultiIndex para tickers individuales, especialmente si se descargan múltiples tickers.
        if isinstance(self.datos.columns, pd.MultiIndex):
            # Crear una nueva lista de nombres de columnas usando el primer nivel del MultiIndex
            new_columns = [col[0] for col in self.datos.columns]
            self.datos.columns = new_columns
            
        #print(self.datos.head())
        #print(self.datos.index) # Esto nos dirá si el índice es de fechas o numérico
        
        rows_to_delete_labels = self.datos.index[[1, 2]].tolist()
        self.data = self.datos.drop(rows_to_delete_labels).copy() # .copy() para crear un nuevo DataFrame
        
        #print('datos convertidos:\n', self.data) 
        
        # Añadir los osciladores como nuevas columnas
        self.data['RSI'] = ta.rsi(self.data['Close'])
        # Asignar la columna 'MACD' del DataFrame resultado a tu DataFrame principal
        
        # Verifica que haya suficientes datos para el cálculo (la ventana por defecto es 26)
        if len(self.data['Close']) > 26:
            # Calcula el MACD
            macd_df = ta.macd(self.data['Close'])
            #print('macd_df:\n', macd_df)

            # Asigna las columnas solo si macd_df no está vacío
            if not macd_df.empty:
                self.data['MACD'] = macd_df['MACD_12_26_9']
                self.data['MACDh'] = macd_df['MACDh_12_26_9']
                self.data['MACDs'] = macd_df['MACDs_12_26_9']
            else:
                # Manejar el caso en que el cálculo de MACD falle
                print("Advertencia: El cálculo de MACD devolvió un DataFrame vacío. No se añadirán las columnas.")
                # O puedes asignar valores nulos para evitar errores posteriores
                self.data['MACD'] = np.nan
                self.data['MACDh'] = np.nan
                self.data['MACDs'] = np.nan
        else:
            # Manejar el caso de datos insuficientes
            print("Advertencia: La serie de datos es demasiado corta para calcular el MACD.")
            self.data['MACD'] = np.nan
            self.data['MACDh'] = np.nan
            self.data['MACDs'] = np.nan

        # Estocastico
        self.data['Stoch_K'] = ta.stoch(self.data['High'], self.data['Low'], self.data['Close'])['STOCHk_14_3_3']
        self.data['Stoch_D'] = ta.stoch(self.data['High'], self.data['Low'], self.data['Close'])['STOCHd_14_3_3']
        
        # ATR
        # Supongamos que tienes un DataFrame con columnas 'high', 'low', 'close'
        self.data['ATR'] = ta.atr(high=self.data['High'], low=self.data['Low'], close=self.data['Close'], length=14)
        
        # ADX
        adx = ta.adx(high=self.data['High'], low=self.data['Low'], close=self.data['Close'], length=14)
        # Unir al DataFrame original Ya que crea varias columnas 
        # ADX_14: fuerza de la tendencia (valor principal del ADX) sin importar si alcista o bajista
        # DMP_14: +DI (Directional Movement Positive) Mide la fuerza alcista
        # DMN_14: –DI (Directional Movement Negative) Mide la fuerza bajista
        self.data = self.data.join(adx)
        
        # OBV El OBV acumula el volumen diario según el movimiento del precio. Precio sube -> Suma Precio baja -> Resta
        # Calcula el OBV directamente
        self.data['OBV'] = ta.obv(close=self.data['Close'], volume=self.data['Volume'])
        
        # Crear el indicador VWAP
        # Calcular VWAP
        self.data.ta.vwap(append=True)
        
        # Media Móvil Simple (SMA): Una media móvil simple (SMA) es un cálculo que toma la media aritmética de un 
        # conjunto dado de precios durante un número específico de periodos en el pasado, como por ejemplo, 
        # durante los últimos 15, 30, 100 o 200 días.
        self.data["SMA"]=ta.sma(self.data["Close"],10)
        
        # Media móvil simple de 20 días
        self.data['SMA_20'] = ta.sma(self.data['Close'], length=20)

        # Media móvil exponencial de 20 días
        self.data['EMA_20'] = ta.ema(self.data['Close'], length=20)
        
        # Añadir el VIX a los datos de cotizacion
        self.data = self.data.merge(self.vix, left_index=True, right_index=True, how='left')
        
        
        
        #print("Numero de columnas:\n", self.data.columns)

        # Es importante eliminar los valores NaN que se generan al inicio de la serie
        # debido a la ventana de cálculo de los osciladores.
        self.data = self.data.dropna()
        self.datos_completos = self.data
        
        #print('datos convertidos:\n', self.data) 

        
        
        
        for widget in self.main_frame.winfo_children():
            widget.destroy()
            
        self.option_graph.set("Cotización") 
        self.cotiza_graph()    
            
            
            
    def cambia_graph(self, event):
        if event.widget.get() == 'Cotización':
            self.cotiza_graph()
        elif event.widget.get() == 'Previsión': 
            self.LSTM_ticker()    
        elif event.widget.get() == 'Osciladores': 
            self.graph_6_ticker()             
        elif event.widget.get() == 'Incongruencias': 
            self.datos_incongruentes()   
        elif event.widget.get() == 'Información': 
            self.informacion_ticker()               
        elif event.widget.get() == 'Análisis': 
            self.analisis_ticker()               
        elif event.widget.get() == 'Recomendaciones': 
            self.recomendacion_graph()               
        elif event.widget.get() == 'Noticias': 
            self.noticias_ticker()    
        else:
            self.datos_ticker() 
            
               
                
    ##############################################################
    ######################### COTIZACIONES #######################
    ##############################################################
    
    def cotiza_graph(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()
        
        # Extraer el precio de cierre
        precio_cierre = self.history['Close'].iloc[0]
    
            
        # Crear figura y eje
        fig, ax = plt.subplots(figsize=(12, 6))

        # Graficar en el eje
        ax.plot(self.data["Close"], label=f"Precio de cierre - {precio_cierre:.2f}", color="blue", linewidth=2)
        ax.plot(self.data["High"], label="Precio máximo", color="green", linewidth=0.5)
        ax.plot(self.data["Low"], label="Precio mínimo", color="red", linewidth=0.5)
        
        # 3. Configurar el Cursor
        # El cursor añadirá líneas verticales y horizontales que siguen el ratón.
        # Establece useblit=True para un mejor rendimiento.
        # color='red' para el color de las líneas.
        #self.cursor = Cursor(ax, horizOn=True, vertOn=True, color='red', linewidth=1.0, useblit=True)
        
        # Asegurarse de que volumen_escalado sea una serie unidimensional
        volumen_escalado = self.data["Volume"] / self.data["Volume"].max()
        
        

        # Eliminar posibles NaNs
        volumen_escalado = volumen_escalado.dropna()

        # Asegurar que los índices coincidan
        x = volumen_escalado.index
        y = volumen_escalado.values

        # Graficar
        #ax.fill_between(x, y, color="gray", alpha=0.3, label="Volumen diario (escalado)")
        ax.fill_between(x, y.flatten(), color="gray", alpha=0.3, label="Volumen diario (escalado)")
        nmb = self.dame_nombre(self.ticker.get())
        ax.set_title(f"Evolución del precio de {self.ticker.get()} - {nmb}")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Precio (€)")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()    
        
        #########################
        
        # 2. CREACIÓN DEL OBJETO ANOTACIÓN (TOOLTIP)
        # Inicialmente se crea invisible.
        self.annot = ax.annotate(
            "Coordenadas",
            xy=(0, 0),
            xytext=(20, 20),  # Desplazamiento del texto (offset) respecto al punto (xy)
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w", alpha=0.8),  # Estilo de la caja (fondo blanco semi-transparente)
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color="k"),
            visible=False
        )

        # 3. FUNCIÓN DE MANEJO DE EVENTOS (CALLBACK)
        def hover(event):
            """Función que se llama cada vez que el ratón se mueve."""
            
            # Comprueba si el evento ocurrió sobre el área de los ejes
            if event.inaxes == ax:
                # Obtener las coordenadas X e Y del cursor en los datos del gráfico
                coord_x_num = event.xdata
                coord_y = event.ydata
                
                # Coordenadas en píxeles de la pantalla (para la lógica de límites)
                coord_x_pix = event.x
                coord_y_pix = event.y
                
                # Obtener el renderer del canvas
                renderer = fig.canvas.get_renderer()
        
                # Obtener la transformación de datos a píxeles del eje (ax)
                transform = ax.transData
                
                # 1. OBTENER LOS LÍMITES DEL EJE EN PÍXELES
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                
                # Convertir los límites del eje (datos) a píxeles
                # [(x_min, y_min), (x_max, y_max)] en píxeles
                min_pix = transform.transform((xlim[0], ylim[0]))
                max_pix = transform.transform((xlim[1], ylim[1]))

                x_min_pix, y_min_pix = min_pix
                x_max_pix, y_max_pix = max_pix

                # Definir el desplazamiento base y el margen de seguridad en píxeles
                offset_x = 15  # Desplazamiento inicial a la derecha
                offset_y = 15  # Desplazamiento inicial arriba
                margen = 100   # Píxeles desde el borde donde debe cambiar el offset
                ha = 'left'   # Horizontal Alignment (Izquierda)
                va = 'bottom' # Vertical Alignment (Abajo)
                
                
                # A. Comprobar el límite DERECHO
                if x_max_pix - coord_x_pix < margen:
                    # Mover el texto a la IZQUIERDA del punto.
                    print("Activando desplazamiento a la IZQUIERDA")
                    offset_x = -15
                    ha = 'right' # Anclar el texto a la derecha del punto
                else:
                    offset_x = 15
                    ha = 'left'  # Anclar el texto a la izquierda del punto
                # B. Comprobar el límite SUPERIOR e INFERIOR
                # Nota: Matplotlib (y muchos backends) tiene Y=0 en la parte INFERIOR
                
                # Comprobar borde SUPERIOR (Y-píxel alto)
                if coord_y_pix > y_max_pix - margen: 
                    # Si está cerca del borde superior, mover el texto ABAJO.
                    offset_y = -15
                    va = 'top' # Anclar el texto en la parte superior del punto
        
                # Comprobar borde INFERIOR (Y-píxel bajo)
                elif coord_y_pix < y_min_pix + margen:
                    # Si está cerca del borde inferior, mover el texto ARRIBA.
                    offset_y = 15    
                    va = 'bottom' # Anclar el texto en la parte inferior del punto            
                else:
                    # Para el centro, podemos dejar el offset y centrar verticalmente si es necesario
                    va = 'center'        
                        
                # 1. Convertir el número decimal de fecha a objeto datetime
                # Es crucial usar num2date y la zona horaria del eje si se especificó,
                # aunque si no se especificó, la conversión simple funciona.
                try:
                    # Convertir a datetime
                    fecha_dt = mdates.num2date(coord_x_num)
            
                    # Formatear la fecha como cadena de texto (ejemplo: '2023-10-17')
                    # Puedes cambiar el formato según necesites: '%Y-%m-%d %H:%M:%S'
                    fecha_str = fecha_dt.strftime('%Y-%m-%d') 
                except ValueError:
                    # En caso de que el cursor esté en una zona sin datos válidos
                    fecha_str = "Fecha no válida"

                # 1. Formatear y actualizar el texto del tooltip
                text = f"Fecha: {fecha_str}\nPrecio: {coord_y:.3f}" # .3f para 3 decimales
                self.annot.set_text(text)

                # 2. Actualizar la posición del tooltip
                # Ajusta la alineación horizontal y vertical
                self.annot.set_ha(ha)
                self.annot.set_va(va)
                
                
                # 3. APLICAR EL NUEVO DESPLAZAMIENTO (OFFSET)
                self.annot.xytext = (offset_x, offset_y) # Ajusta el desplazamiento del texto
                # Esto establece el punto (xy) al que apunta la flecha
                self.annot.xy = (coord_x_num, coord_y)

                # 3. Hacer visible el tooltip
                self.annot.set_visible(True)

                # 4. Redibujar la figura para mostrar los cambios
                # draw_idle() es mejor para eventos de movimiento que el draw() normal
                fig.canvas.draw_idle()
            else:
                # Si el ratón está fuera de los ejes, ocultar el tooltip
                if self.annot.get_visible():
                    self.annot.set_visible(False)
                    fig.canvas.draw_idle()

        # 4. CONEXIÓN DEL EVENTO
        # 'motion_notify_event' se dispara cada vez que el puntero se mueve sobre el lienzo.
        fig.canvas.mpl_connect("motion_notify_event", hover)
                
        #########################
        
        
        
        
        
        # Insertar figura en Tkinter
        self.canvas = FigureCanvasTkAgg(fig, master=self.main_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)        
       
        # Finalmente dibuja
        self.canvas.draw_idle()            
        
        
        
    ##############################################################
    ########################## DATOS #############################
    ##############################################################
        
    def datos_ticker(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy() 
            
        if self.data is None or self.data.empty:
            mb.showinfo("Información", "No hay datos para mostrar.")
            return             
            
            
        # --- Crear un Frame contenedor para el Treeview y sus Scrollbars ---
        treeview_container = ttk.Frame(self.main_frame)
        treeview_container.pack(fill="both", expand=True, padx=5, pady=5) # Este frame se expande

        # Crear el Treeview dentro del contenedor
        treeRes = ttk.Treeview(treeview_container, columns=list(self.data.columns), show="headings")  

        style = ttk.Style()
        style.configure("Treeview", rowheight=25)
        # Colores alternados
        style.map("Treeview", background=[("selected", "#3399FF")])  # Color al seleccionar
        treeRes.tag_configure("evenrow", background="#e4e4e9")
        treeRes.tag_configure("oddrow", background="#ffffff")
        
  
        
        # Configurar encabezados y columnas
        for col in self.data.columns:
            treeRes.heading(col, text=col)
            # Puedes ajustar el ancho de las columnas aquí si lo deseas
            treeRes.column(col, anchor="center", width=150) 

        # Insertar filas
        for i, (idx, row) in enumerate(self.data.iterrows()):
            # Incluir el índice (fecha) como la primera columna visible si es relevante
            # O puedes insertar solo los valores de las columnas
            tag = "evenrow" if i % 2 == 0 else "oddrow"
            treeRes.insert("", "end", values=(row.tolist()), tags=(tag,))    
            
        # Empaquetar el Treeview DENTRO del contenedor
        # side="left" es crucial para dejar espacio para el scrollbar vertical
        # .. treeRes.pack(side="left", fill="both", expand=True)

        # Scrollbars (también dentro del contenedor)
        scrollRes_y = ttk.Scrollbar(treeview_container, orient="vertical", command=treeRes.yview)
        scrollRes_x = ttk.Scrollbar(treeview_container, orient="horizontal", command=treeRes.xview)

        # Empaquetar scrollbars al lado del Treeview
        # .. scrollRes_y.pack(side="right", fill="y")
        # .. scrollRes_x.pack(side="bottom", fill="x") # Scrollbar horizontal abajo del Treeview

        treeRes.configure(yscrollcommand=scrollRes_y.set, xscrollcommand=scrollRes_x.set)    
        
        # Usar grid para organizar los widgets
        treeRes.grid(row=0, column=0, sticky="nsew")
        scrollRes_y.grid(row=0, column=1, sticky="ns")
        scrollRes_x.grid(row=1, column=0, sticky="ew")

        # Configurar el contenedor para que se expanda correctamente
        treeview_container.rowconfigure(0, weight=1)
        treeview_container.columnconfigure(0, weight=1)       
            
        
    ##############################################################
    ####################### INCONGRUENCIAS #######################
    ##############################################################
        
    def datos_incongruentes(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()  
            
        if self.data is None or self.data.empty:
            mb.showinfo("Información", "No hay datos para buscar incongruencias. Realice una búsqueda primero.")
            return

        columnas = ['High', 'Close', 'Low']
        faltantes = [col for col in columnas if col not in self.data.columns]
        if faltantes:
            mb.showerror("Error de Datos", f"Error: faltan las columnas {faltantes} en los datos descargados para buscar incongruencias.")
            return

        # Asegurarse de que las columnas sean numéricas antes de la comparación
        for col in columnas:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

        df_clean = self.data.dropna(subset=columnas)
        incongruentes = df_clean[(df_clean['High'] < df_clean['Close']) | 
                                 (df_clean['Close'] < df_clean['Low']) | 
                                 (df_clean['High'] <= df_clean['Low'])
                                ]   
        
        if incongruentes.empty:
            mb.showinfo("Información", "No se encontraron incongruencias en los datos.")
            return

        # --- Crear un Frame contenedor para el Treeview y sus Scrollbars ---
        treeview_container = ttk.Frame(self.main_frame)
        treeview_container.pack(fill="both", expand=True, padx=5, pady=5) # Este frame se expande

        # Crear el Treeview dentro del contenedor
        treeRes = ttk.Treeview(treeview_container, columns=list(incongruentes.columns), show="headings")    
        
        # Configurar encabezados y columnas
        for col in incongruentes.columns:
            treeRes.heading(col, text=col)
            # treeRes.column(col, anchor="center", width=100) # Ajustar anchos si es necesario

        # Insertar filas
        for idx, row in incongruentes.iterrows():
            # Puedes usar idx (la fecha) como el primer elemento si quieres mostrarla
            treeRes.insert("", "end", values=(row.tolist()))
            
        # Empaquetar el Treeview DENTRO del contenedor
        treeRes.pack(side="left", fill="both", expand=True)

        # Scrollbars (también dentro del contenedor)
        scrollRes_y = ttk.Scrollbar(treeview_container, orient="vertical", command=treeRes.yview)
        scrollRes_x = ttk.Scrollbar(treeview_container, orient="horizontal", command=treeRes.xview)

        scrollRes_y.pack(side="right", fill="y")
        scrollRes_x.pack(side="bottom", fill="x") # Horizontal abajo del Treeview

        treeRes.configure(yscrollcommand=scrollRes_y.set, xscrollcommand=scrollRes_x.set)                        
        return    
        
    ##############################################################
    ######################### INFORMACION ########################
    ##############################################################    

    def informacion_ticker(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()  
            
        if self.info is None:
            mb.showinfo("Información", "No hay información del ticker para mostrar. Realice una búsqueda primero.")
            return   
        
        # Diccionario de campos de interés y sus traducciones
        fields_to_translate = {
            "titulo1": "Información general",
            "shortName": "Nombre Corto",
            "longName": "Nombre Largo",
            "address1": "Dirección",
            "city": "Ciudad",
            "zip": "Código Postal",
            "country": "País",
            "website": "Sitio Web",
            "industry": "Industria",
            "sector": "Sector",
            "longBusinessSummary": "Resumen de Negocio (Largo)",
            "fullTimeEmployees": "Empleados a Tiempo Completo",
            "exchange": "Bolsa",
            "currency": "Moneda",
            "financialCurrency": "Moneda financiera",
            "grossMargins": "Márgenes Brutos",
            "freeCashflow": "Flujo de Caja Libre",
            "titulo2": "Información financiera",
            "marketCap": "Capitalización de Mercado",
            "enterpriseValue":"Valor empresarial",
            "totalRevenue": "Ingresos Totales",
            "netIncomeToCommon": "Beneficio neto",
            "profitMargins": "Márgenes de Beneficio",
            "returnOnAssets": "Retorno sobre Activos",
            "returnOnEquity": "Retorno sobre Patrimonio Neto ROE (Alto: Eficiencia/Negativo: Perdidas)",
            "totalCash": "Efectivo Total",
            "totalDebt": "Deuda Total",
            "bookValue": "Valor Contable",
            "priceToBook": "Precio a Valor Contable",
            "operatingCashflow": "Flujo de Caja Operativo",
            "earningsGrowth": "Crecimiento de Ganancias",
            "revenueGrowth": "Crecimiento de Ingresos",
            "operatingMargins": "Margen operativo",
            "grossProfits": "Beneficio bruto",
            "titulo3": "Datos del mercado",
            "currentPrice": "Precio Actual",
            "regularMarketPreviousClose": "Cierre Anterior Mercado Regular",
            "regularMarketOpen": "Apertura Mercado Regular",
            "regularMarketDayHigh": "Máximo Día Mercado Regular",
            "regularMarketDayLow": "Mínimo Día Mercado Regular",            
            "regularMarketVolume": "Volumen Mercado Regular",
            "averageVolume": "Volumen promedio",
            "trailingPE": "PER histórico",
            "forwardPE": "PER proyectado",
            "trailingEps": "BPA histórico",
            "forwardEps": "BPA proyectado",
            "beta": "Beta (Volatilidad)",
            "fiftyTwoWeekHigh": "Máximo 52 Semanas",
            "fiftyTwoWeekLow": "Mínimo 52 Semanas",
            "fiftyDayAverage": "Media 50 Días",
            "dividendRate": "Tasa de dividendo",
            "dividendYield": "Rentabilidad por Dividendo",
            "payoutRatio": "Ratio de Payout",
            "lastDividendValue": "Último dividendo pagado",
            "lastDividendDate": "Fecha del último dividendo",
            "revenuePerShare": "Ingresos por Acción",
            "quickRatio": "Ratio de Liquidez (Quick Ratio)",
            "currentRatio": "Ratio de Liquidez (Current Ratio)",
            "bid": "Oferta de Compra (Bid)",
            "ask": "Oferta de Venta (Ask)",
            "twoHundredDayAverage": "Media 200 Días",
            "targetHighPrice": "Precio Objetivo Máximo",
            "targetLowPrice": "Precio Objetivo Mínimo",
            "targetMeanPrice": "Precio Objetivo Medio",
            "recommendationKey": "Recomendación Clave",
            
            "titulo4": "Gobierno Corporativo y Riesgos",
            "auditRisk": "Riesgo de Auditoría",
            "boardRisk": "Riesgo de Junta Directiva",
            "compensationRisk": "Riesgo de Compensación",
            "shareHolderRightsRisk": "Riesgo de Derechos de Accionistas",
            "overallRisk": "Riesgo General",
        }

        # --- Crear un Frame contenedor para el Treeview y sus Scrollbars ---
        treeview_container = ttk.Frame(self.main_frame)
        treeview_container.pack(fill="both", expand=True, padx=5, pady=5) # Este frame se expande

        # Crear el Treeview dentro del contenedor
        self.tree_info = ttk.Treeview(treeview_container, columns=("Atributo", "Valor"), show="headings")
        self.tree_info.heading("Atributo", text="Atributo", anchor="w")
        self.tree_info.heading("Valor", text="Valor", anchor="w")
        
        # Ajustar anchos de columna (pueden ser dinámicos o fijos, pero el 'Valor' debe ser amplio)
        self.tree_info.column("Atributo", width=250, anchor="w") # Suficiente para la traducción
        self.tree_info.column("Valor", width=650, anchor="w") # Amplio para descripciones largas
        
        # Estilo para los encabezados
        self.tree_info.tag_configure("encabezado", font=('TkDefaultFont', 14, 'bold')) 
        self.tree_info.tag_configure("relevante", foreground="red") 

        # Empaquetar el Treeview DENTRO del contenedor
        # side="left" es crucial para dejar espacio para el scrollbar vertical
        self.tree_info.pack(side="left", fill="both", expand=True)
        
        style = ttk.Style()
        style.configure("Treeview", rowheight=25)
        # Colores alternados
        style.map("Treeview", background=[("selected", "#3399FF")])  # Color al seleccionar
        self.tree_info.tag_configure("evenrow", background="#e4e4e9")
        self.tree_info.tag_configure("oddrow", background="#ffffff")

        # Scrollbars (también dentro del contenedor)
        scroll_info_y = ttk.Scrollbar(treeview_container, orient="vertical", command=self.tree_info.yview)
        scroll_info_y.pack(side="right", fill="y")
        # No necesitas scroll_info_x si tus columnas son fijas y no esperas un desbordamiento horizontal

        self.tree_info.configure(yscrollcommand=scroll_info_y.set)
        
        # Usar un diccionario para almacenar los datos completos para el pop-up
        self.full_info_data = {} 
        h = True

        for key, display_name in fields_to_translate.items():
            value = self.info.get(key, "N/D") 
            tag = "evenrow" if h % 2 == 0 else "oddrow"
               
            if key == "longBusinessSummary" and isinstance(value, str):
                # Traducir el resumen completo
                translated_summary = self.translator(value, dest='es')
                self.full_info_data[key] = translated_summary # Guardar el texto completo traducido

                # Mostrar un fragmento en el Treeview para que sea manejable
                short_summary = textwrap.shorten(translated_summary, width=90, placeholder="...")
                self.tree_info.insert("", "end", values=(display_name, short_summary), tags=(tag,), iid=key) # Usamos 'key' como iid
            elif key in ["lastDividendDate", "lastSplitDate"]:
                timestamp = value
                
                # Zona horaria de Barcelona
                zona_barcelona = pytz.timezone("Europe/Madrid")

                # Convertir a fecha con zona horaria
                try:
                    fecha_local = datetime.fromtimestamp(timestamp, zona_barcelona)
                    fecha_legible = fecha_local.strftime("%A %d de %B de %Y, %H:%M:%S") 
                except:
                    fecha_legible = 'N/D'                   
                self.tree_info.insert("", "end", values=(display_name, fecha_legible))
                    
            elif isinstance(value, (int, float)):
                if key in ["marketCap", "regularMarketVolume", "totalCash", "totalDebt", "totalRevenue", "freeCashflow", "operatingCashflow",
                            "enterpriseValue","netIncomeToCommon","grossProfits", "averageVolume"]:
                    formatted_value = f"{value:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".") 
                    self.tree_info.insert("", "end", values=(display_name, formatted_value), tags=(tag,), iid=key)
                else:
                    self.tree_info.insert("", "end", values=(display_name, value), tags=(tag,), iid=key)
            else:
                if key == "titulo1":
                    self.tree_info.insert("", "end", values=("Información general", ""), tags=("encabezado",))
                elif key == "titulo2":
                    self.tree_info.insert("", "end", values=("Información financiera", ""), tags=("encabezado",))    
                elif key == "titulo3":
                    self.tree_info.insert("", "end", values=("Datos del Mercado", ""), tags=("encabezado",))    
                elif key == "titulo4":
                    self.tree_info.insert("", "end", values=("Gobierno corporativo y riesgos", ""), tags=("encabezado",))    

                else:    
                    self.tree_info.insert("", "end", values=(display_name, value), tags=(tag,), iid=key)
            
            h = False if h else True
            
            
        # Bind para abrir el pop-up cuando se selecciona una fila
        self.tree_info.bind("<<TreeviewSelect>>", self.on_tree_info_select)

    def on_tree_info_select(self, event):
        selected_item_id = self.tree_info.focus() # Obtener el ID del elemento seleccionado
        if selected_item_id:
            # Obtener los valores de la fila seleccionada
            item_values = self.tree_info.item(selected_item_id, 'values')
            if item_values and item_values[0] == "Resumen de Negocio (Largo)": # Si es la fila del resumen
                # Usar el iid (que es la clave original 'longBusinessSummary') para obtener el texto completo
                full_summary = self.full_info_data.get(selected_item_id) 
                if full_summary:
                    self.mostrar_resumen_completo(full_summary)

    def mostrar_resumen_completo(self, text_content):
        top = tk.Toplevel(self)
        top.title("Resumen de Negocio Completo")
        top.geometry("700x400") # Tamaño de la ventana emergente

        # Crear un ScrolledText para mostrar el contenido largo
        text_widget = st.ScrolledText(top, wrap="word", font=("Arial", 10), width=80, height=20)
        text_widget.pack(fill="both", expand=True, padx=10, pady=10)
        text_widget.insert(tk.END, text_content)
        text_widget.config(state="disabled") # Hacer el texto de solo lectura

    def translator(self, text, dest='es'):
        try:
            # La librería deep_translator necesita que el texto no esté vacío
            if not text.strip():
                return ""
            return GoogleTranslator(source='auto', target=dest).translate(text)
        except Exception as e:
            print(f"Error en la traducción: {e}")
            return text # Retorna el texto original en caso de error
        
    ##############################################################
    ######################### ANALISIS ###########################
    ##############################################################
      
    def formato_numero(self, numero):
        try:
            float(numero)
            formateado = f"{numero:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except:
            formateado = "0"    
            
        return formateado
      
    def analisis_ticker(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy() 
        
        if self.financials.empty:
            print(f"No se pudieron obtener los datos de la cuenta de resultados para {self.ticker}.")
            return None             
            
        if self.financials is None:
            mb.showinfo("Información", "No hay información del ticker para mostrar. Realice una búsqueda primero.")
            return
        
       
        ########################################
        
        def on_mousewheel(event):
            #print("Evento desde:", type(event.widget))
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            

        # Canvas + Scrollbar
        canvas = tk.Canvas(self.main_frame, height=300)
        scrollbar = tk.Scrollbar(self.main_frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Frame interno dentro del Canvas
        frame_interno = tk.Frame(canvas)
        canvas.create_window((0, 0), window=frame_interno, anchor="nw")

        # Función para actualizar el scroll
        def actualizar_scroll(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        frame_interno.bind("<Configure>", actualizar_scroll)
        
        # Vincular la rueda del ratón
        canvas.bind_all("<MouseWheel>", on_mousewheel)  # Para Windows y Linux
        
        # Configurar columnas: 7 columnas con ancho mínimo de 100 píxeles
        for col in range(6):
            frame_interno.columnconfigure(col, minsize=150)
        
        lt = self.financials.iloc[:, 0] # Toma la primera columna (datos más recientes)
        print('lt:\n', lt)
        
        self.fn = Financials()
        self.fn.set_completa_clase(lt)
        self.fn.set_sector(self.info["sector"])
        self.fn.set_genera_partidas()
        
        # Vamos a tomar los datos trimestrales
        quarterly = self.quarterly_financials
        
        self.qf = Quarterly_financials()
        self.qf.set_sector(self.info["sector"])
        self.qf.set_completa_clase(quarterly)
        
        #pd.set_option('display.max_rows', None)
        #print('lt', lt)
        fila = 0
        # Crear widgets en tres columnas
        tk.Label(frame_interno, text=f"Cuenta de Resultados", fg='crimson', font=("Arial", 14, "bold")
                ).grid(row=0, column=0, columnspan=2)
        # Crear widgets en tres columnas
        tk.Label(frame_interno, text=f"Ultimos cinco trimestres", fg='crimson', font=("Arial", 14, "bold")
                ).grid(row=0, column=2, columnspan=5)
        fila += 1
        
        tk.Label(frame_interno, text='Partida', fg="blue", font=("Arial", 11, "bold")
                 ).grid(row=fila, column=0)
        tk.Label(frame_interno, text='Importe', fg="blue", font=("Arial", 11, "bold")
                 ).grid(row=fila, column=1)
        tk.Label(frame_interno, text=self.qf.get_nombre_columna(0), fg="blue", font=("Arial", 11, "bold")
                 ).grid(row=fila, column=2)
        tk.Label(frame_interno, text=self.qf.get_nombre_columna(1), fg="blue", font=("Arial", 11, "bold")
                 ).grid(row=fila, column=3)
        tk.Label(frame_interno, text=self.qf.get_nombre_columna(2), fg="blue", font=("Arial", 11, "bold")
                 ).grid(row=fila, column=4)
        tk.Label(frame_interno, text=self.qf.get_nombre_columna(3), fg="blue", font=("Arial", 11, "bold")
                 ).grid(row=fila, column=5)
        tk.Label(frame_interno, text=self.qf.get_nombre_columna(4), fg="blue", font=("Arial", 11, "bold")
                 ).grid(row=fila, column=6)

        fila += 1
        
        # Ventas
        tk.Label(frame_interno, text=f"Ventas").grid(row=fila, column=0)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.fn.get_ventas())}").grid(row=fila, column=1)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_ventas(0))}").grid(row=fila, column=2)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_ventas(1))}").grid(row=fila, column=3)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_ventas(2))}").grid(row=fila, column=4)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_ventas(3))}").grid(row=fila, column=5)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_ventas(4))}").grid(row=fila, column=6)

        fila += 1
        # Coste de ventas
        tk.Label(frame_interno, text=f"Coste de Ventas", width=30).grid(row=fila, column=0)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.fn.get_coste_ventas())}").grid(row=fila, column=1)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_coste_ventas(0))}").grid(row=fila, column=2)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_coste_ventas(1))}").grid(row=fila, column=3)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_coste_ventas(2))}").grid(row=fila, column=4)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_coste_ventas(3))}").grid(row=fila, column=5)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_coste_ventas(4))}").grid(row=fila, column=6)
        fila += 1
        # Beneficio Bruto
        tk.Label(frame_interno, text=f"Márgen bruto", width=30).grid(row=fila, column=0, padx=5, pady=1)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.fn.get_beneficio_bruto())}").grid(row=fila, column=1)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_beneficio_bruto(0))}").grid(row=fila, column=2)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_beneficio_bruto(1))}").grid(row=fila, column=3)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_beneficio_bruto(2))}").grid(row=fila, column=4)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_beneficio_bruto(3))}").grid(row=fila, column=5)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_beneficio_bruto(4))}").grid(row=fila, column=6)
        
        fila += 1
        
        tk.Label(frame_interno, text=f"Gastos operativos").grid(row=fila, column=0)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.fn.get_gastos_operativos())}").grid(row=fila, column=1)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_gastos_operativos(0))}").grid(row=fila, column=2)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_gastos_operativos(1))}").grid(row=fila, column=3)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_gastos_operativos(2))}").grid(row=fila, column=4)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_gastos_operativos(3))}").grid(row=fila, column=5)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_gastos_operativos(4))}").grid(row=fila, column=6)

        fila += 1
        tk.Label(frame_interno, text=f"BAII - Rdo antes de intereses e Imptos.").grid(row=fila, column=0)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.fn.get_BAII())}").grid(row=fila, column=1)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_BAII(0))}").grid(row=fila, column=2)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_BAII(1))}").grid(row=fila, column=3)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_BAII(2))}").grid(row=fila, column=4)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_BAII(3))}").grid(row=fila, column=5)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_BAII(4))}").grid(row=fila, column=6)

        fila += 1
        tk.Label(frame_interno, text=f"Gastos financieros").grid(row=fila, column=0)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.fn.get_gastos_financieros())}").grid(row=fila, column=1)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_gastos_financieros(0))}").grid(row=fila, column=2)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_gastos_financieros(1))}").grid(row=fila, column=3)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_gastos_financieros(2))}").grid(row=fila, column=4)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_gastos_financieros(3))}").grid(row=fila, column=5)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_gastos_financieros(4))}").grid(row=fila, column=6)
        
        fila += 1
        tk.Label(frame_interno, text=f"BAI - Resultado antes de Imptos.").grid(row=fila, column=0)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.fn.get_BAI())}").grid(row=fila, column=1)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_BAI(0))}").grid(row=fila, column=2)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_BAI(1))}").grid(row=fila, column=3)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_BAI(2))}").grid(row=fila, column=4)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_BAI(3))}").grid(row=fila, column=5)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_BAI(4))}").grid(row=fila, column=6)

        fila += 1
        tk.Label(frame_interno, text=f"Impuestos").grid(row=fila, column=0)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.fn.get_impuestos())}").grid(row=fila, column=1)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_impuestos(0))}").grid(row=fila, column=2)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_impuestos(1))}").grid(row=fila, column=3)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_impuestos(2))}").grid(row=fila, column=4)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_impuestos(3))}").grid(row=fila, column=5)
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_impuestos(4))}").grid(row=fila, column=6)

        fila += 1
        tk.Label(frame_interno, text=f"Beneficio Neto", fg="darkgreen", font=("Arial", 10, "bold")
                 ).grid(row=fila, column=0)
        color = 'darkgreen' if self.fn.get_beneficio_neto() >= 0 else 'red'
        tk.Label(frame_interno, text=f"{self.formato_numero(self.fn.get_beneficio_neto())}", fg=color, font=("Arial", 10, "bold")
                 ).grid(row=fila, column=1)
        color = 'darkgreen' if self.qf.get_beneficio_neto(0) >= 0 else 'red'
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_beneficio_neto(0))}", fg=color, font=("Arial", 10, "bold")
                 ).grid(row=fila, column=2)
        color = 'darkgreen' if self.qf.get_beneficio_neto(1) >= 0 else 'red'
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_beneficio_neto(1))}", fg=color, font=("Arial", 10, "bold")
                 ).grid(row=fila, column=3)
        color = 'darkgreen' if self.qf.get_beneficio_neto(2) >= 0 else 'red'
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_beneficio_neto(2))}", fg=color, font=("Arial", 10, "bold")
                 ).grid(row=fila, column=4)
        color = 'darkgreen' if self.qf.get_beneficio_neto(3) >= 0 else 'red'
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_beneficio_neto(3))}", fg=color, font=("Arial", 10, "bold")
                 ).grid(row=fila, column=5)
        color = 'darkgreen' if self.qf.get_beneficio_neto(4) >= 0 else 'red'
        tk.Label(frame_interno, text=f"{self.formato_numero(self.qf.get_beneficio_neto(4))}", fg=color, font=("Arial", 10, "bold")
                 ).grid(row=fila, column=6)


        fila += 1
        
        
        ##############################################################################################
        ######################################### Ratios #############################################
        ##############################################################################################
                
        if self.balance_sheet.empty:
            print(f"No se pudieron obtener los datos financieros para {self.ticker}.")
            return None             
            
        if self.balance_sheet is None:
            mb.showinfo("Información", "No hay información del ticker para mostrar. Realice una búsqueda primero.")
            return
        
        balan = self.balance_sheet.iloc[:, 0] # Toma la primera columna (datos más recientes)
        
        self.bs = Balance()
        self.bs.set_completa_clase(balan)
        
        fila = fila + 2
        # Crear widgets en tres columnas
        tk.Label(frame_interno, text=f"Ratios", fg='crimson', font=("Arial", 14, "bold")
                ).grid(row=fila, column=0, columnspan=2)
        fila += 1
        
        ########################## Liquidez ##########################33
        
        tk.Label(frame_interno, text='Ratios de liquidez', fg="blue", font=("Arial", 11, "bold")
                 ).grid(row=fila, column=0)
        fila += 1
        
        # Calcular el ratio
        #ratio_liquidez_corriente(activo_corriente, pasivo_corriente)
        activo_corriente = self.bs.get_activo_corriente()
        pasivo_corriente = self.bs.get_pasivo_corriente()
        try:
            liquidez_corriente  = activo_corriente / pasivo_corriente
        except:
            liquidez_corriente  = 0
                
        tk.Label(frame_interno, text=f"Activo corriente").grid(row=fila, column=0)
        tk.Label(frame_interno, text=f"{self.formato_numero(activo_corriente)}").grid(row=fila, column=1)
        fila += 1
        tk.Label(frame_interno, text=f"Pasivo corriente").grid(row=fila, column=0)
        tk.Label(frame_interno, text=f"{self.formato_numero(pasivo_corriente)}").grid(row=fila, column=1)
        tk.Label(frame_interno, text="Activo corriente / Pasivo corriente", anchor="w",fg='darkslategray'
                ).grid(row=fila, columnspan=5, column=2, sticky='w')
        fila += 1
          
        color = 'red' if liquidez_corriente < 1 else 'darkgreen' if liquidez_corriente > 1.5 else 'darkorange'

        tk.Label(frame_interno, text="Liquidez corriente", fg=color, font=("Arial", 11, "bold")
                ).grid(row=fila, column=0)
        tk.Label(frame_interno, text=f"{self.formato_numero(liquidez_corriente)}", fg=color, font=("Arial", 11, "bold")
                ).grid(row=fila, column=1)
        tk.Label(frame_interno, text="< 1 → Problemas de liquidez | 1 → Justo | > 1.5 → Buena liquidez", anchor="w"
                ).grid(row=fila, columnspan=5, column=2, sticky='w')
        fila += 1
        
        inventario = self.bs.get_inventario()
        tk.Label(frame_interno, text="Inventario").grid(row=fila, column=0)
        tk.Label(frame_interno, text=f"{self.formato_numero(inventario)}").grid(row=fila, column=1)
        tk.Label(frame_interno, text="(Activo corriente - Inventario) / Pasivo corriente", fg='darkslategray'
                ).grid(row=fila, columnspan=5, column=2, sticky='w')
        fila += 1
        try:
            prueba_acida = (activo_corriente - inventario) / pasivo_corriente
        except:
            prueba_acida = 0
        color = 'red' if prueba_acida < 1 else 'darkgreen' if prueba_acida > 1.5 else 'darkorange'        
        tk.Label(frame_interno, text="Prueba ácida/ Quick ratio", fg=color, font=("Arial", 11, "bold")
                ).grid(row=fila, column=0)
        tk.Label(frame_interno, text=f"{self.formato_numero(prueba_acida)}", fg=color, font=("Arial", 11, "bold")
                ).grid(row=fila, column=1)
        tk.Label(frame_interno, text="Más estricto que la liquidez corriente").grid(row=fila, columnspan=5, column=2, sticky='w')
        fila += 1
        
        # Ratio de tesoreria
        efectivo = self.bs.get_efectivo()
        
        tk.Label(frame_interno, text=f"Efectivo").grid(row=fila, column=0)
        tk.Label(frame_interno, text=f"{self.formato_numero(efectivo)}").grid(row=fila, column=1)
        tk.Label(frame_interno, text="Efectivo / Pasivo corriente", fg='darkslategray'
                ).grid(row=fila, columnspan=5, column=2, sticky='w')
        fila += 1
        try:
            tesoreria = efectivo / pasivo_corriente
        except:
            tesoreria = 0
            
        color = 'red' if tesoreria < 1 else 'darkgreen' if tesoreria > 1 else 'darkorange'        
        tk.Label(frame_interno, text="Tesorería", fg=color, font=("Arial", 11, "bold")).grid(row=fila, column=0)
        tk.Label(frame_interno, text=f"{self.formato_numero(tesoreria)}", fg=color, font=("Arial", 11, "bold")
                ).grid(row=fila, column=1)
        tk.Label(frame_interno, text="> 1 → Capacidad de pago inmediata | = 1 → Equilibrio justo | < 1 → Riesgo financiero"
                ).grid(row=fila, columnspan=5, column=2, sticky='w')
        fila += 1
        
        ################# Rentabilidad ####################3
        
        # ROA 
        tk.Label(frame_interno, text='Ratios de rentabilidad', fg="blue", font=("Arial", 11, "bold")
                ).grid(row=fila, column=0)
        fila += 1
        # ROA - Rentabilidad sobre activos
        beneficio_neto = self.fn.get_beneficio_neto()
        tk.Label(frame_interno, text="Beneficio Neto").grid(row=fila, column=0)
        tk.Label(frame_interno, text=f"{self.formato_numero(beneficio_neto)}").grid(row=fila, column=1)
        tk.Label(frame_interno, text="Beneficio neto (X-1) / Activo corriente", fg='darkslategray'
                 ).grid(row=fila, columnspan=5, column=2, sticky='w')
        fila += 1
        try:
            roa = beneficio_neto / activo_corriente
        except:
            roa = 0
        color = 'red' if roa < 1 else 'darkgreen' if roa >= 5 else 'darkorange'         
        tk.Label(frame_interno, text="ROA - Rentabilidad sobre activos", fg=color, font=("Arial", 11, "bold")
                 ).grid(row=fila, column=0)
        tk.Label(frame_interno, text=f"{self.formato_numero(roa)}", fg=color, font=("Arial", 11, "bold")
                 ).grid(row=fila, column=1)
        tk.Label(frame_interno, text="> 10 → Excelente | 5 a 10 → Buena eficiencia | 1 a 5 → Aceptable | < 1 → Baja rentabilidad", 
                anchor="w").grid(row=fila, columnspan=5, column=2, sticky='w')
        fila += 1        
        # ROE - Rentabilidad sobre patrimonio
        patrimonio_neto = self.bs.get_patrimonio_neto() # Patrimonio de los accionistas
        tk.Label(frame_interno, text="Patrimonio Neto").grid(row=fila, column=0)
        tk.Label(frame_interno, text=f"{self.formato_numero(patrimonio_neto)}").grid(row=fila, column=1)
        tk.Label(frame_interno, text="(Beneficio neto / Patrimonio neto) * 100", fg='darkslategray'
                ).grid(row=fila, columnspan=5, column=2, sticky='w')
        fila += 1
        try:
            roe = beneficio_neto / patrimonio_neto * 100
        except:
            roe = 0
        color = 'red' if roe < 5 else 'darkgreen' if roe >= 10 else 'darkorange'     
        tk.Label(frame_interno, text="ROE - Rentabilidad sobre patrimonio", fg=color, font=("Arial", 11, "bold")
                 ).grid(row=fila, column=0)
        tk.Label(frame_interno, text=f"{self.formato_numero(roe)}", fg=color, font=("Arial", 11, "bold")
                 ).grid(row=fila, column=1)
        tk.Label(frame_interno, text="> 15 → Excelente | 10 a 15 → Buena eficiencia | 5 a 10 → Aceptable | < 5 → Baja rentabilidad", 
                 anchor="w").grid(row=fila, columnspan=5, column=2, sticky='w')
        fila += 1        
        # Margen neto
        ingresos_totales = self.fn.get_ventas() # Ventas
        tk.Label(frame_interno, text="Ingresos totales").grid(row=fila, column=0)
        tk.Label(frame_interno, text=f"{self.formato_numero(patrimonio_neto)}").grid(row=fila, column=1)
        tk.Label(frame_interno, text="(Beneficio neto / Ingresos totales) * 100", fg='darkslategray'
                 ).grid(row=fila, columnspan=5, column=2, sticky='w')
        fila += 1
        try:
            margen_neto = beneficio_neto / ingresos_totales * 100
        except:
            margen_neto = 0
        color = 'red' if margen_neto < 5 else 'darkgreen' if margen_neto >= 10 else 'darkorange'         
        tk.Label(frame_interno, text="Márgen Neto", width=30, fg=color, font=("Arial", 11, "bold")
                 ).grid(row=fila, column=0, padx=5, pady=1)
        tk.Label(frame_interno, text=f"{self.formato_numero(margen_neto)}", width=30, fg=color, font=("Arial", 11, "bold")
                 ).grid(row=fila, column=1, padx=5, pady=1)
        tk.Label(frame_interno, text="> 20 → Excelente | 10 a 20 → Buena | 5 a 10 → Moderada | < 5 → Baja rentabilidad", 
                 anchor="w").grid(row=fila, columnspan=5, column=2, sticky='w')
        fila += 1   
        # Margen operativo
        beneficio_operativo = self.fn.get_resultado_operativo() 
        tk.Label(frame_interno, text="Beneficio operativo").grid(row=fila, column=0)
        tk.Label(frame_interno, text=f"{self.formato_numero(beneficio_operativo)}").grid(row=fila, column=1)
        tk.Label(frame_interno, text="(Beneficio operativo / Ingresos totales) * 100", fg='darkslategray'
                 ).grid(row=fila, columnspan=5, column=2, sticky='w')
        fila += 1
        try:
            margen_operativo = beneficio_operativo / ingresos_totales * 100
        except:
            margen_operativo = 0
            
        color = 'red' if margen_operativo < 5 else 'darkgreen' if margen_operativo >= 10 else 'darkorange'         
        tk.Label(frame_interno, text="Márgen Operativo", fg=color, font=("Arial", 11, "bold")
                 ).grid(row=fila, column=0)
        tk.Label(frame_interno, text=f"{self.formato_numero(margen_operativo)}", fg=color, font=("Arial", 11, "bold")
                 ).grid(row=fila, column=1)
        tk.Label(frame_interno, text="> 20 → Excelente | 10 a 20 → Buena | 5 a 10 → Moderada | < 5 → Débil rentabilidad", 
                 anchor="w").grid(row=fila, columnspan=5, column=2, sticky='w')
        fila += 1   
        
        ######################################################################
        ########################### Endeudamiento ############################   
        ######################################################################
        tk.Label(frame_interno, text='Endeudamiento', fg="blue", font=("Arial", 11, "bold")
                 ).grid(row=fila, column=0)
        fila += 1
        # Ratio de endeudamiento
        activo_total = self.bs.get_activo_total() 
        pasivo_total = self.bs.get_pasivo_total() 
        
        tk.Label(frame_interno, text="Activo Total").grid(row=fila, column=0)
        tk.Label(frame_interno, text=f"{self.formato_numero(activo_total)}").grid(row=fila, column=1)
        fila += 1
        tk.Label(frame_interno, text="Pasivo Total").grid(row=fila, column=0)
        tk.Label(frame_interno, text=f"{self.formato_numero(pasivo_total)}").grid(row=fila, column=1)
        tk.Label(frame_interno, text="Pasivo total / Activo total", fg='darkslategray'
                 ).grid(row=fila, columnspan=5, column=2, sticky='w')
        fila += 1
        try:
            ratio_endeudamiento = pasivo_total / activo_total
        except:
            ratio_endeudamiento = 0
        color = 'red' if ratio_endeudamiento > 0.7 else 'darkgreen' if ratio_endeudamiento < 0.5 else 'darkorange'         
        tk.Label(frame_interno, text="Ratio de endeudamiento", fg=color, font=("Arial", 11, "bold")
                 ).grid(row=fila, column=0)
        tk.Label(frame_interno, text=f"{self.formato_numero(ratio_endeudamiento)}", fg=color, font=("Arial", 11, "bold")
                 ).grid(row=fila, column=1)
        tk.Label(frame_interno, text="< 0.5 → Buena salud financiera | 0.5 a 0.7 → Moderada | > 0.7 → Posible riesgo financiero"
                 ).grid(row=fila, columnspan=5, column=2, sticky='w')
        fila += 1  
        
        # Ratio de Deuda a Patrimonio 
        tk.Label(frame_interno, text="Pasivo total / Patrimonio neto", fg='darkslategray'
                 ).grid(row=fila, columnspan=5, column=2, sticky='w')
        fila += 1
        try:
            ratio_deuda = pasivo_total / patrimonio_neto
        except:
            ratio_deuda = 0
        color = 'red' if ratio_deuda > 1 else 'darkgreen' if ratio_deuda < 1 else 'darkorange'        
        tk.Label(frame_interno, text="Ratio de Deuda a patrimonio", fg=color, font=("Arial", 11, "bold")
                 ).grid(row=fila, column=0)
        tk.Label(frame_interno, text=f"{self.formato_numero(ratio_deuda)}", fg=color, font=("Arial", 11, "bold")
                 ).grid(row=fila, column=1)
        tk.Label(frame_interno, text="< 1 → Menor riesgo financiero | = 1 → Equilibrada | > 1 → Mayor riesgo financiero"
                 ).grid(row=fila, columnspan=5, column=2, sticky='w')
        fila += 1  
        
        # Cobertura de Intereses
        ebit = self.fn.get_BAII() 
        gasto_intereses = self.fn.get_gastos_financieros() 
        
        tk.Label(frame_interno, text="EBIT").grid(row=fila, column=0)
        tk.Label(frame_interno, text=f"{self.formato_numero(ebit)}").grid(row=fila, column=1)
        fila += 1
        tk.Label(frame_interno, text="Gasto por Intereses").grid(row=fila, column=0)
        tk.Label(frame_interno, text=f"{self.formato_numero(gasto_intereses)}").grid(row=fila, column=1)
        tk.Label(frame_interno, text="EBIT / Gastos por Intereses", fg='darkslategray'
                 ).grid(row=fila, columnspan=5, column=2, sticky='w')
        fila += 1
        try:
            cobertura_intereses = ebit / gasto_intereses
        except:
            cobertura_intereses = 0
        color = 'red' if cobertura_intereses < 1 else 'darkgreen' if cobertura_intereses > 2 else 'darkorange'        
        tk.Label(frame_interno, text="Cobertura de intereses", fg=color, font=("Arial", 11, "bold")
                 ).grid(row=fila, column=0)
        tk.Label(frame_interno, text=f"{self.formato_numero(cobertura_intereses)}", fg=color, font=("Arial", 11, "bold")
                 ).grid(row=fila, column=1)
        tk.Label(frame_interno, text="< 1 → No cubre intereses | 1 a 2 → Riesgo moderado | > 2 → Buena capacidad | > 5 → Excelente solvencia", 
                 ).grid(row=fila, columnspan=5, column=2, sticky='w')
        fila += 1  
        
        ######################################################################
        ########################### Eficiencia  ##############################   
        ######################################################################
        tk.Label(frame_interno, text='Eficiencia', fg="blue", font=("Arial", 11, "bold")
                 ).grid(row=fila, column=0)
        fila += 1
        # Rotación de activos
        tk.Label(frame_interno, text="Ingresos Totales / Activo Total",  fg='darkslategray'
                 ).grid(row=fila, columnspan=5, column=2, sticky='w')
        fila += 1
        try:
            rotacion_activos = ingresos_totales / activo_total
        except:
            rotacion_activos = 0
        color = 'red' if rotacion_activos < 1 else 'darkgreen' if rotacion_activos > 2 else 'darkorange' 
        tk.Label(frame_interno, text="Rotación de activos", fg=color, font=("Arial", 11, "bold")
                 ).grid(row=fila, column=0)
        tk.Label(frame_interno, text=f"{self.formato_numero(rotacion_activos)}", fg=color, font=("Arial", 11, "bold")
                 ).grid(row=fila, column=1)
        tk.Label(frame_interno, text="> 2 → Alta eficiencia. Convierte rapidamente activos en ventas  | 1 a 2 → Aceptable | < 1 → Baja", 
                 ).grid(row=fila, columnspan=5, column=2, sticky='w')
        fila += 1  
        # Rotación de inventario
        coste_ventas = self.fn.get_coste_ventas()
        # Inventario promedio (usamos dos años si están disponibles)
        try:
            inventory_current = self.balance_sheet.loc["Inventory"].iloc[0]
            inventory_previous = self.balance_sheet.loc["Inventory"].iloc[1]
            inventory_avg = (inventory_current + inventory_previous) / 2
        except:
            inventory_current = 0
            inventory_previous = 0
            inventory_avg = 0
                
        tk.Label(frame_interno, text="Coste de Ventas").grid(row=fila, column=0)
        tk.Label(frame_interno, text=f"{self.formato_numero(coste_ventas)}").grid(row=fila, column=1)
        fila += 1
        tk.Label(frame_interno, text="Inventario ultimo ejercicio").grid(row=fila, column=0)
        tk.Label(frame_interno, text=f"{self.formato_numero(inventory_current)}").grid(row=fila, column=1)
        fila += 1
        tk.Label(frame_interno, text="Inventario ejercicio anterior").grid(row=fila, column=0)
        tk.Label(frame_interno, text=f"{self.formato_numero(inventory_previous)}").grid(row=fila, column=1)
        fila += 1
        tk.Label(frame_interno, text="Inventario medio").grid(row=fila, column=0)
        tk.Label(frame_interno, text=f"{self.formato_numero(inventory_avg)}").grid(row=fila, column=1)
        tk.Label(frame_interno, text="Coste de Ventas / Inventario Medio", fg='darkslategray'
                 ).grid(row=fila, columnspan=5, column=2, sticky='w')
        fila += 1
        try:
            rotacion_inventario = coste_ventas / inventory_avg
        except:
            rotacion_inventario = 0
        color = 'red' if rotacion_inventario < 1 else 'darkgreen' if rotacion_inventario < 10 else 'darkorange'         
        tk.Label(frame_interno, text="Rotación de inventario", fg=color, font=("Arial", 11, "bold")
                 ).grid(row=fila, column=0)
        tk.Label(frame_interno, text=f"{self.formato_numero(rotacion_inventario)}", fg=color, font=("Arial", 11, "bold")
                 ).grid(row=fila, column=1)
        tk.Label(frame_interno, 
                 text="> 10 veces año → Muy alta (Riesgo rotura stocks) | 6 a 10 → Alta (Demanda saludable | < 1 → Baja", 
                 ).grid(row=fila, columnspan=5, column=2, sticky='w')
        fila += 1  
        # Rotación de Cuentas por Cobrar
                
        fila += 1
        # Crear widgets en tres columnas
        tk.Label(frame_interno, text="------------- Fin del Análisis ----------------", fg='crimson', font=("Arial", 14, "bold")
                ).grid(row=fila, column=0, columnspan=2)
               
        
    ##############################################################################################
    ######################################## RECOMENDACIONES #####################################
    ##############################################################################################
    
    def recomendacion_graph(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()
        target_price = self.info.get("targetMeanPrice")
        real_price = self.info.get("currentPrice")
        rm = self.recomendaciones.copy()    
        

        # Extraer la primera fila como valores escalares
        fila = rm.head(1).copy()
        fila["targetPrice"] = target_price
        fila["realPrice"] = real_price

        period = fila["period"].iloc[0]
        recs = [
            fila["strongBuy"].iloc[0],
            fila["buy"].iloc[0],
            fila["hold"].iloc[0],
            fila["sell"].iloc[0],
            fila["strongSell"].iloc[0]
        ]
        precios = [fila["targetPrice"].iloc[0], fila["realPrice"].iloc[0]]
        labels = ["Compra fuerte", "Compra", "Mantener", "Vender", "Venta fuerte"]
        precio_labels = ["Precio objetivo", "Precio real"]

        x_recs = range(len(recs))
        x_precio = [len(recs) // 2]  # posición centrada para los precios
        
        # Crear figura y ejes
        fig, ax1 = plt.subplots(figsize=(8, 5))
        
        # Colores dinámicos
        color_map = ["green", "lime", "orange", "red", "darkred"]
        colors = color_map[:len(recs)]

        # Gráfico de barras
        ax1.bar(x_recs, recs, color=colors)
        ax1.set_xticks(x_recs)
        ax1.set_xticklabels(labels[:len(recs)], rotation=45)
        ax1.set_ylabel("Cantidad de recomendaciones", color="darkred")
        ax1.tick_params(axis='y', labelcolor="darkred")

        # Segundo eje para precios
        ax2 = ax1.twinx()
        

        
        # 💰 Segundo eje: precios con símbolos grandes y etiquetas
        ax2.scatter(x_precio, [precios[0]], color="blue", s=200, marker="^", label=f"Precio objetivo: {target_price:.2f} ")
        ax2.scatter(x_precio, [precios[1]], color="red", s=200, marker="D", label=f"Precio real: {real_price:.2f} ")
        ax2.set_ylabel("Precio ($)", color="blue")
        ax2.tick_params(axis='y', labelcolor="blue")

        # Título y leyenda
        fig.suptitle(f"Recomendaciones y Precios en el momento actual")
        ax2.legend(loc="upper right", fontsize=14)
                    
        fig.tight_layout()

        # Insertar figura en Tkinter
        self.canvas = FigureCanvasTkAgg(fig, master=self.main_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)        
       
        # Finalmente dibuja
        self.canvas.draw_idle()     
        
     
    ##################################################################################
    #################### PANTALLA DE 6 GRAFICOS #######################################  
    ##################################################################################
              
    def graph_6_ticker(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()  
            
            
            
        def on_mousewheel(event):
            #print("Evento desde:", type(event.widget))
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            

        # Canvas + Scrollbar
        #..canvas = tk.Canvas(self.main_frame, height=300)
        canvas = tk.Canvas(self.main_frame)
        scrollbar = tk.Scrollbar(self.main_frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Frame interno dentro del Canvas    
        frame_interno = tk.Frame(canvas)
        

        # Función para actualizar el scroll
        def actualizar_scroll(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        frame_interno.bind("<Configure>", actualizar_scroll)
        canvas.create_window((0, 0), window=frame_interno, anchor="nw")
        
        # Vincular la rueda del ratón
        #canvas.bind_all("<MouseWheel>", on_mousewheel)  # Para Windows y Linux 
        canvas.bind("<MouseWheel>", on_mousewheel)  # Para Windows y Linux           
            
        osciladores = self.data[['Close','High','Low','RSI','MACD','MACDh','MACDs','Stoch_K','Stoch_D','ATR','OBV','ADX_14','DMP_14','DMN_14','VIX','VWAP_D']].copy()
        
        # Vamos a añadir las señales de compra y de venta
        # Asegúrate de tener los indicadores calculados en 'osciladores'
        
        
        # Opcion 1
        #..print('Oscilaores:\n', osciladores)
        

        # Inicializamos la columna de señales
        osciladores['sg_RSI']   = ''
        osciladores['sg_MACD']  = ''
        osciladores['sg_Stock'] = ''
        osciladores['sg_ATR']   = ''
        osciladores['sg_OBV']   = ''
        osciladores['sg_ADX']   = ''
        osciladores['sg_VIX']   = ''
        osciladores['sg_VWAP']   = ''
        
        # Creamos las funciones para actualizar los valores de los diferentes osciladores
        def señal_rsi(rsi):
            if rsi < 30:
                return 'compra'
            elif rsi > 70:
                return 'venta'
            return 'J'

        def señal_macd(prev_macd, prev_signal, curr_macd, curr_signal):
            if prev_macd < prev_signal and curr_macd > curr_signal:
                return 'compra'
            elif prev_macd > prev_signal and curr_macd < curr_signal:
                return 'venta'
            return ''

        def señal_stoch(k, d, prev_k, prev_d):
            if k > d and prev_k <= prev_d:
                return 'compra'
            elif k < d and prev_k >= prev_d:
                return 'venta'
            return ''

        def señal_atr(atr):
            return 'V' if atr > 1.5 else 'N/V'

        def señal_obv(close, prev_close, obv, prev_obv):
            if close > prev_close and obv > prev_obv:
                return 'V'
            elif close < prev_close and obv < prev_obv:
                return 'D'
            return '-'

        def señal_adx(dmp, dmn, adx):
            if dmp > dmn:
                if adx > 75: return '⇑⇑'
                elif adx > 50: return '⇑'
                elif adx > 25: return '↑'
                elif adx > 20: return '↗'
                else: return '→'
            elif dmp < dmn:
                if adx > 75: return '⇓⇓'
                elif adx > 50: return '⇓'
                elif adx > 25: return '↓'
                elif adx > 20: return '↘'
                else: return '←'
            return '⇔'
        
        def señal_vix(vix):
            return ''
        
        def señal_vwap(vwap):
            return ''


        # Vamos a poner valor a los campos de señal
        for i in range(1, len(osciladores)):
            fecha_actual = osciladores.index[i]
            fecha_anterior = osciladores.index[i - 1]

            osciladores.loc[fecha_actual, 'sg_RSI'] = señal_rsi(osciladores.loc[fecha_actual, 'RSI'])

            osciladores.loc[fecha_actual, 'sg_MACD'] = señal_macd(
                osciladores.loc[fecha_anterior, 'MACD'], osciladores.loc[fecha_anterior, 'MACDs'],
                osciladores.loc[fecha_actual, 'MACD'], osciladores.loc[fecha_actual, 'MACDs']
            )

            osciladores.loc[fecha_actual, 'sg_Stock'] = señal_stoch(
                osciladores.loc[fecha_actual, 'Stoch_K'], osciladores.loc[fecha_actual, 'Stoch_D'],
                osciladores.loc[fecha_anterior, 'Stoch_K'], osciladores.loc[fecha_anterior, 'Stoch_D']
            )

            osciladores.loc[fecha_actual, 'sg_ATR'] = señal_atr(osciladores.loc[fecha_actual, 'ATR'])

            osciladores.loc[fecha_actual, 'sg_OBV'] = señal_obv(
                osciladores.loc[fecha_actual, 'Close'], osciladores.loc[fecha_anterior, 'Close'],
                osciladores.loc[fecha_actual, 'OBV'], osciladores.loc[fecha_anterior, 'OBV']
            )

            osciladores.loc[fecha_actual, 'sg_ADX'] = señal_adx(
                osciladores.loc[fecha_actual, 'DMP_14'], osciladores.loc[fecha_actual, 'DMN_14'],
                osciladores.loc[fecha_actual, 'ADX_14']
            )
            osciladores.loc[fecha_actual, 'sg_VIX'] = señal_vix(osciladores.loc[fecha_actual, 'VIX'])
            osciladores.loc[fecha_actual, 'sg_VWAP'] = señal_vwap(osciladores.loc[fecha_actual, 'VWAP_D'])
            


        #..print('Oscilaores antes:\n', osciladores[['sg_RSI', 'sg_MACD', 'sg_Stock', 'sg_ATR', 'sg_OBV', 'sg_ADX']])
        osciladores.dropna(inplace=True)
            
        # Tomar los ultimos 30 dìas
        osc_30 = osciladores.tail(30)
        
        #print('Oscilaores:\n', osciladores)
        
        print('Osc 30:\n', osc_30[['MACD','sg_RSI', 'sg_MACD', 'sg_Stock', 'sg_ATR', 'sg_OBV', 'sg_ADX','sg_VIX', 'sg_VWAP']])

        
        
            
        
            
        # Suponiendo que 'osciladores' ya tiene las columnas:
        # 'RSI', 'MACD', 'MACD_signal', 'Stoch_K', 'Stoch_D'
        # sharex=True hace que todos los graficos compartan el mismo eje x
        
        # Crear una fuente personalizada
        fuente_leyenda = FontProperties(family='Arial', style='italic', size=9)
        
        
        fig, axs = plt.subplots(8, 1, figsize=(12, 18), sharex=True)
        # Ajusta la altura entre gráficos para poder ver las leyendas
        plt.subplots_adjust(hspace=0.5, top=0.92)


        fig.suptitle('Indicadores Técnicos: RSI, MACD, Estocástico, ATR, OBV, ADX, VIX, VWAP', fontsize=16)

        # --- RSI ---
        axs[0].plot(osc_30.index, osc_30['RSI'], label='RSI', color='blue')
        axs[0].axhline(70, color='red', linestyle='--', linewidth=1)
        axs[0].axhline(30, color='green', linestyle='--', linewidth=1)
        axs[0].axhline(50, color='darkseagreen', linestyle='--', linewidth=2)
        
        axs[0].set_ylabel('RSI')
        #axs[0].legend('')
        axs[0].legend(loc='upper left', prop=fuente_leyenda)
        axs[0].grid(True)
        # Texto debajo del gráfico RSI
        axs[0].text(0.5, -0.25, 'Indicador RSI: mide la fuerza relativa del precio, RSI < 30 -> Comprar | RSI > 70 -> Vender', 
                    transform=axs[0].transAxes, ha='center', fontsize=10
                    )
        
        # Señal de compra o venta
        for i in range(1, len(osc_30)):
            fecha = osc_30.index[i]
            rsi_val = osc_30.loc[fecha, 'RSI']
            señal = osc_30.loc[fecha, 'sg_RSI']

            if señal == 'compra':
                axs[0].annotate('Compra',
                    xy=(fecha, rsi_val),
                    xytext=(fecha, rsi_val + 5),
                    arrowprops=dict(facecolor='green', shrink=0.05),
                    fontsize=9, color='green')

            elif señal == 'venta':
                axs[0].annotate('Venta',
                    xy=(fecha, rsi_val),
                    xytext=(fecha, rsi_val - 5),
                    arrowprops=dict(facecolor='red', shrink=0.05),
                    fontsize=9, color='red')

        
        # --- Colores condicionales para las barras ---
        colors = ['green' if val >= 0 else 'red' for val in osc_30['MACDh']]

        # --- Gráfico MACD ---
        axs[1].plot(osc_30.index, osc_30['MACD'], label='MACD', color='darkgreen')
        axs[1].plot(osc_30.index, osc_30['MACDs'], label='Línea de señal', color='lightgreen', linestyle='--')

        # --- Histograma como barras ---
        axs[1].bar(osc_30.index, osc_30['MACDh'], color=colors, alpha=0.5, label='Histograma')

        # --- Línea horizontal en 0 ---
        axs[1].axhline(0, color='gray', linestyle='--', linewidth=1)

        # --- Estética ---
        axs[1].set_ylabel('MACD')
        axs[1].legend(loc='upper left', prop=fuente_leyenda)
        axs[1].grid(True)
        axs[1].text(0.5, -0.25, 'MACD cruza por encima de línea de señal -> Compra | MACD cruza por debajo de línea de señal -> Venta', 
                    transform=axs[1].transAxes, ha='center', fontsize=10)

        # --- Flechas de señal ---
        ymin, ymax = axs[1].get_ylim()
        rango_y = ymax - ymin
        desplazamiento = 0.05 * rango_y

        for i in range(1, len(osc_30)):
            fecha = osc_30.index[i]
            macd_val = osc_30.loc[fecha, 'MACD']
            señal = osc_30.loc[fecha, 'sg_MACD']

            if señal == 'compra':
                axs[1].annotate('Compra',
                    xy=(fecha, macd_val),
                    xytext=(fecha, macd_val + desplazamiento),
                    arrowprops=dict(arrowstyle='fancy', color='green', mutation_scale=30),
                    fontsize=9, color='green')

            elif señal == 'venta':
                axs[1].annotate('Venta',
                    xy=(fecha, macd_val),
                    xytext=(fecha, macd_val + desplazamiento),
                    arrowprops=dict(arrowstyle='fancy', color='red', mutation_scale=20),
                    fontsize=9, color='red')

        # --- Estocástico ---
        axs[2].plot(osc_30.index, osc_30['Stoch_K'], label='%K', color='purple')
        axs[2].plot(osc_30.index, osc_30['Stoch_D'], label='%D', color='violet', linestyle='--')
        axs[2].axhline(80, color='red', linestyle='--', linewidth=1)
        axs[2].axhline(20, color='green', linestyle='--', linewidth=1)
        axs[2].set_ylabel('Estocástico')
        axs[2].legend(loc='upper left', prop=fuente_leyenda)
        axs[2].grid(True)
        axs[2].text(0.5, -0.25, 'Stoch_K < 20 y cruza por encima de Stoch_D -> Compra | '
                    'Stoch_K > 80 y cruza por debajo de Stoch_D -> Venta',
                    transform=axs[2].transAxes, ha='center', fontsize=10
                    )
        ymin, ymax = axs[2].get_ylim()
        rango_y = ymax - ymin
        desplazamiento = 0.05 * rango_y  # flecha de 5% del alto del gráfico
        
        for i in range(1, len(osc_30)):
            fecha = osc_30.index[i]
            stoch_val = osc_30.loc[fecha, 'Stoch_K']
            señal = osc_30.loc[fecha, 'sg_Stock']

            if señal == 'compra':
                axs[2].annotate('Compra',
                    xy=(fecha, stoch_val),
                    xytext=(fecha, stoch_val + desplazamiento),
                    arrowprops=dict(arrowstyle='fancy', color='green', mutation_scale=30),
                    fontsize=9, color='green')

            elif señal == 'venta':
                axs[2].annotate('Venta',
                    xy=(fecha, stoch_val),
                    xytext=(fecha, stoch_val + desplazamiento),
                    arrowprops=dict(arrowstyle='fancy', color='red', mutation_scale=20),
                    fontsize=9, color='red')

        # --- ATR ---
        axs[3].plot(osc_30.index, osc_30['ATR'], label='Volatilidad', color='darkorange', linewidth=2)
        axs[3].set_ylabel('ATR')
        axs[3].legend(loc='upper left', prop=fuente_leyenda)
        axs[3].grid(True)
        axs[3].text(0.5, -0.25, 'El ATR se expresa en unidades de precio | '
                    'ATR sube → el activo está más volátil | ATR baja → el activo está más estable.',
                    transform=axs[3].transAxes, ha='center', fontsize=10
                    )
        
        # Eje Y derecho: Volatilidad
        ax3 = axs[3].twinx()
        ax3.plot(osc_30.index, osc_30['Close'], color='#00CED1', label='Precio Acción', linestyle='--', linewidth=0.5)
        ax3.set_ylabel('Precio Acción', color='#00CED1')
        ax3.tick_params(axis='y', labelcolor='#00CED1')
        
        # --- OBV ---
        axs[4].plot(osc_30.index, osc_30['OBV'], label='OBV', color='brown', linewidth=2)
        axs[4].set_ylabel('OBV')
        axs[4].legend(loc='upper left', prop=fuente_leyenda)
        axs[4].grid(True)
        axs[4].text(0.5, -0.25, 'OBV creciente → presión de compra → posible continuación alcista | '
                    'OBV decreciente → presión de venta → posible continuación bajista.',
                    transform=axs[4].transAxes, ha='center', fontsize=10
                    )
        axs[4].yaxis.set_major_formatter(mattk.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
        
            # Eje Y derecho: Volatilidad
        ax4 = axs[4].twinx()
        ax4.plot(osc_30.index, osc_30['Close'], color='#00CED1', label='Precio Acción', linestyle='--', linewidth=0.5)
        ax4.set_ylabel('Precio Acción', color='#00CED1')
        ax4.tick_params(axis='y', labelcolor='#00CED1')

        # --- ADX ---
        axs[5].plot(osc_30.index, osc_30['ADX_14'], label='ADX: Tendencia', color='#8B864E', linewidth=2)
        axs[5].plot(osc_30.index, osc_30['DMP_14'], label='+DI: Presión alcista', color='#CD9B1D', linestyle='--')
        axs[5].plot(osc_30.index, osc_30['DMN_14'], label='-DI: Presión bajista', color='#FFD700', linestyle='--')
        axs[5].set_ylabel('ADX')
        axs[5].legend(loc='upper left', prop=fuente_leyenda)
        axs[5].grid(True)
        axs[5].text(0.5, -0.25, '< 20 → Tendencia débil o lateral | 20–40 → Tendencia moderada | > 40 → Tendencia fuerte'
                    ' | +DI > -DI → Tendencia alcista | -DI > +DI → Tendencia bajista',
                    transform=axs[5].transAxes, ha='center', fontsize=10
                    )
        ax5 = axs[5].twinx()
        ax5.plot(osc_30.index, osc_30['Close'], color='#00CED1', label='Precio Acción', linestyle='--', linewidth=0.5)
        ax5.set_ylabel('Precio Acción', color='#00CED1')
        ax5.tick_params(axis='y', labelcolor='#00CED1')
        ymin, ymax = axs[5].get_ylim()
        rango_y = ymax - ymin
        desplazamiento = 0.05 * rango_y  # flecha de 5% del alto del gráfico
        
        for i in range(1, len(osc_30)):
            fecha = osc_30.index[i]
            atr_val = osc_30.loc[fecha, 'ADX_14']
            señal = osc_30.loc[fecha, 'sg_ADX']

            axs[5].annotate(señal,
                xy=(fecha, atr_val),
                xytext=(fecha, atr_val + desplazamiento),
                fontsize=9, color='green')


        # --- VIX ---
        axs[6].plot(osc_30.index, osc_30['VIX'], label='VIX', color='#2F4F4F', linewidth=2)
        axs[6].set_ylabel('VIX')
        axs[6].axhline(20, color='red', linestyle='--', alpha=0.8, label='Zona de Miedo (> 20)')
        axs[6].legend(loc='upper left', prop=fuente_leyenda)
        axs[6].grid(True)
        axs[6].text(0.5, -0.25, 'El VIX mide si en el Mercado se respira un sentimiento optimista | '
                    'Por debajo de 20 se muestra un sentimiento pesimista.',
                    transform=axs[6].transAxes, ha='center', fontsize=10
                    )
        
        # Eje Y derecho: Volatilidad
        ax6 = axs[6].twinx()
        ax6.plot(osc_30.index, osc_30['Close'], color='#00CED1', label='Precio Acción', linestyle='--', linewidth=0.5)
        ax6.set_ylabel('Precio Acción', color='#00CED1')
        ax6.tick_params(axis='y', labelcolor='#00CED1')

        # --- VWAP ---
        axs[7].plot(osc_30.index, osc_30['VWAP_D'], label='VWAP', color='#FF1493', linewidth=2)
        axs[7].set_ylabel('VWAP')
        axs[7].legend(loc='upper left', prop=fuente_leyenda)
        axs[7].grid(True)
        axs[7].text(0.5, -0.25, 'Precio por encima del VWAP → Compradores dominan → Tendencia alcista | '
                    'Precio por debajo del VWAP → Vendedores dominan → Tendencia bajista.',
                    transform=axs[7].transAxes, ha='center', fontsize=10
                    )
        #axs[4].yaxis.set_major_formatter(mattk.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
        
            # Eje Y derecho: Volatilidad
        ax7 = axs[7].twinx()
        ax7.plot(osc_30.index, osc_30['Close'], color='#00CED1', label='Precio Acción', linestyle='--', linewidth=1)
        ax7.set_ylabel('Precio Acción', color='#00CED1')
        ax7.tick_params(axis='y', labelcolor='#00CED1')

        axs[0].set_title('RSI: Fuerza Relativa', fontsize=12, fontweight='bold', color='darkgreen')
        axs[1].set_title('MACD: Cruce de Medias', fontsize=12, fontweight='bold', color='darkblue')
        axs[2].set_title('Estocástico: %K vs %D', fontsize=12, fontweight='bold', color='purple')
        axs[3].set_title('ATR: Volatilidad', fontsize=12, fontweight='bold', color='darkorange')
        axs[4].set_title('OBV: presión de compra y venta acumulada a través del volumen', fontsize=12, fontweight='bold', color='brown')
        axs[5].set_title('ADX: Muestra la fuerza de una tendencia', fontsize=12, fontweight='bold', color='#8B864E')
        axs[6].set_title('VIX: Sentimiento del mercado', fontsize=12, fontweight='bold', color='#2F4F4F')
        axs[7].set_title('VWAP: Precio promedio ponderado por volumen. Mide si se está comprando o vendiendo a buen precio.', fontsize=12, fontweight='bold', color='#FF1493')
        

        # --- Eje X ---
        #plt.xlabel('Fecha')
        plt.tight_layout(rect=[0, 0.03, 1, 1])
         
        
        # Insertar figura en Tkinter
        figure_canvas = FigureCanvasTkAgg(fig, master=frame_interno)
        figure_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)        
       
        # Finalmente dibuja
        figure_canvas.draw_idle() 

        
    #############################################################################################
    ################################### NOTICIAS ################################################
    #############################################################################################       
        
    def noticias_ticker(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()  
            
        if self.info is None:
            mb.showinfo("Información", "No hay Noticias del ticker para mostrar. Realice una búsqueda primero.")
            return   
        
        # --- Crear un Frame contenedor para el Treeview y sus Scrollbars ---
        treeview_container = ttk.Frame(self.main_frame)
        treeview_container.pack(fill="both", expand=True, padx=5, pady=5) # Este frame se expande

        # Crear el Treeview dentro del contenedor
        self.tree_news = ttk.Treeview(treeview_container, columns=("Atributo", "Valor"), show="headings")
        self.tree_news.heading("Atributo", text="Concepto", anchor="w")
        self.tree_news.heading("Valor", text="Descripción", anchor="w")
        
        # Ajustar anchos de columna (pueden ser dinámicos o fijos, pero el 'Valor' debe ser amplio)
        self.tree_news.column("Atributo", width=100, anchor="w") # Suficiente para la traducción
        self.tree_news.column("Valor", width=750, anchor="w") # Amplio para descripciones largas
        
        # Estilo para los encabezados
        self.tree_news.tag_configure("encabezado", font=('TkDefaultFont', 14, 'bold')) 
        self.tree_news.tag_configure("relevante", foreground="red") 
        
        style = ttk.Style()
        style.configure("Treeview", rowheight=25)
        # Colores alternados
        style.map("Treeview", background=[("selected", "#3399FF")])  # Color al seleccionar
        self.tree_news.tag_configure("evenrow", background="#e4e4e9")
        self.tree_news.tag_configure("oddrow", background="#ffffff")   
        
             
        
        
        # Empaquetar el Treeview DENTRO del contenedor
        # side="left" es crucial para dejar espacio para el scrollbar vertical
        self.tree_news.pack(side="left", fill="both", expand=True)

        # Scrollbars (también dentro del contenedor)
        scroll_news_y = ttk.Scrollbar(treeview_container, orient="vertical", command=self.tree_news.yview)
        scroll_news_y.pack(side="right", fill="y")
        # No necesitas scroll_info_x si tus columnas son fijas y no esperas un desbordamiento horizontal

        self.tree_news.configure(yscrollcommand=scroll_news_y.set)
        
        # Iterar sobre la lista de diccionarios de noticias
        h=True
        for i, item in enumerate(self.news):
            tag = "evenrow" if h % 2 == 0 else "oddrow"
            noticia = item.get('content', {})
            titulo = noticia.get('title', 'N/A')
            titulo_sp = self.translator(titulo, dest='es')
            fuente = noticia.get('provider', {}).get('displayName', 'N/A')
            enlace = noticia.get('canonicalUrl', {}).get('url', 'N/A')
            fecha_str = noticia.get('pubDate', '')
            
            self.tree_news.insert("", "end", values=(f"Noticia #{i + 1}", ""), tags=("encabezado", tag))
            self.tree_news.insert("", "end", values=('Título', titulo), tags=(tag,))
            self.tree_news.insert("", "end", values=('Traducción', titulo_sp), tags=(tag,))
            self.tree_news.insert("", "end", values=('Fuente', fuente), tags=(tag,))
            self.tree_news.insert("", "end", values=('Publicado', fecha_str), tags=(tag,))
            self.tree_news.insert("", "end", values=('Enlace', enlace), tags=(tag,))
            h = False if h else True
        
        # Bind para abrir el pop-up cuando se selecciona una fila
        self.tree_news.bind("<<TreeviewSelect>>", self.on_tree_news_select)
    
    def on_tree_news_select(self, event):
        selected_item_id = self.tree_news.focus() # Obtener el ID del elemento seleccionado
        if selected_item_id:
            # Obtener los valores de la fila seleccionada
            item_values = self.tree_news.item(selected_item_id, 'values')
            if item_values and item_values[0] == "Enlace": # Si es la fila del enlace
                enlazar = item_values[1]
                webbrowser.open(enlazar)
     

    ###############################################################################
    ####################### MODELO DE PREDICCION LSTM #############################
    ###############################################################################
    
    def LSTM_ticker(self):  
        
        if self.info is None:
            mb.showinfo("Información", "No hay Noticias del ticker para mostrar. Realice una búsqueda primero.")
            return   
        
        # Vamos a montar el frame
        for widget in self.main_frame.winfo_children():
            widget.destroy()
            
        # Vamos a montar los datos del modelo
        self.LSTM_frame = tk.Frame(self.main_frame, bg="dark slate gray", width=1275, height=600)
        self.LSTM_frame.place(x=10, y=10)
        
        # Vamos a montar el frame para la grafica de prevision
        self.LSTM_grafico = tk.Frame(self.LSTM_frame, bg="dark slate gray", width=1275, height=600)
        self.LSTM_grafico.place(x=40, y=180)
        
        # Vamos a montar la barra de progreso
        style = ttk.Style()
        style.theme_use('clam')  # usar tema base
        style.configure("White.TFrame", background="dark slate gray")
        # vamos a crear un frame para colocar el progreso
        self.frame_progreso = ttk.Frame(self.LSTM_frame, width=340, height=120, style="White.TFrame")
        self.frame_progreso.place(x=700, y=10)  
        
        style.configure("MiEstilo.TLabel", background="dark slate gray", foreground="white", font=("Arial", 12, 'bold'))
        
        self.barra_txt_label = ttk.Label(self.frame_progreso, text="", style="MiEstilo.TLabel")
        self.barra_txt_label.place(x=10, y=10)
        # Ponemos estilos a la barra de progreso

        
        style.configure("Modelo.Horizontal.TProgressbar", 
                        troughcolor='dark slate gray', 
                        background='white'
                        )
        
        

        self.barra = ttk.Progressbar(self.frame_progreso, 
                                    orient="horizontal", 
                                    length=300, 
                                    mode="determinate",
                                    style="Modelo.Horizontal.TProgressbar")
        self.barra.place(x=10, y=40)
        # Establecer el rango de la barra
        self.barra["maximum"] = 100
        self.barra["value"] = 0
        
        # Colocar el tiempo de entrenamiento
        style.configure('Tiempo.TLabel', 
                            background='dark slate gray', 
                            foreground='white',
                            font=('Arial', 12, 'bold'))

        self.lb_tiempo_entreno = ttk.Label(self.LSTM_frame, 
                                        text="", 
                                        width=50, 
                                        anchor='e', 
                                        style='Tiempo.TLabel'
                                )
        
        
        self.lb_tiempo_entreno.place(x=800, y=145)
        #self.barra_txt_label.config(text="Texto de la barra")
        #self.lb_tiempo_entreno.config(text="Tiempo de entreno")
        
        # Vamos a colocar
        nmb = self.dame_nombre(self.ticker.get())
        tk.Label(self.LSTM_frame, text=f"Título: {nmb}", bg="dark slate gray", fg='white', font=("Arial", 14, "bold")
                ).place(x=10, y=10)
        
        
        # Precio de compra
        lb_LSTM_precio = self.crear_y_posicionar(
            tk.Label,
            master=self.LSTM_frame,
            place_args={'x': 10, 'y': 60},
            text="Precio:",
            font=("Arial", 10, "bold"),
            anchor="center",
            bg= 'dark slate gray', 
            fg="white",  
            width=13
        )    

        Tooltip(lb_LSTM_precio, " Ultimo precio de cierre." 
        )
        
        self.LSTM_precio = tk.StringVar()
        tk.Entry(self.LSTM_frame, textvariable=self.LSTM_precio, width=12).place(x=130, y=60)
        try:
            mi_precio = float(self.info.get("currentPrice"))
            self.LSTM_precio.set(f"{mi_precio:.2f}")
        except:
            self.LSTM_precio.set("N/A") 
        
        
        
        # Comision de compra
        lb_LSTM_com_compra = self.crear_y_posicionar(
            tk.Label,
            master=self.LSTM_frame,
            place_args={'x': 230, 'y': 60},
            text="Com. Compra:",
            font=("Arial", 10, "bold"),
            anchor="center", 
            bg= 'dark slate gray', 
            fg="white", 
            width=13
        )    

        Tooltip(lb_LSTM_com_compra, " Importe de la comisión bancaria a la compra del título." 
        )
        
        self.LSTM_com_compra = tk.StringVar()
        tk.Entry(self.LSTM_frame, textvariable=self.LSTM_com_compra, width=10).place(x=350, y=60)
        self.LSTM_com_compra.set("10")    

        # Comision de venta
        lb_LSTM_com_venta = self.crear_y_posicionar(
            tk.Label,
            master=self.LSTM_frame,
            place_args={'x': 450, 'y': 60},
            text="Com. Venta:",
            font=("Arial", 10, "bold"),
            anchor="center", 
            bg= 'dark slate gray', 
            fg="white",
            width=13
        )    

        Tooltip(lb_LSTM_com_venta, " Importe de la comisión bancaria a la venta del título." 
        )
        
        self.LSTM_com_venta = tk.StringVar()
        tk.Entry(self.LSTM_frame, textvariable=self.LSTM_com_venta, width=10).place(x=570, y=60)
        self.LSTM_com_venta.set("10")    
        
        # Inversion
        lb_LSTM_inversion = self.crear_y_posicionar(
            tk.Label,
            master=self.LSTM_frame,
            place_args={'x': 10, 'y': 85},
            text="Inversión:",
            font=("Arial", 10, "bold"),
            anchor="center", 
            bg= 'dark slate gray', 
            fg="white",
            width=13
        )    

        Tooltip(lb_LSTM_inversion, " Cantidad a invertir." 
        )
        
        self.LSTM_inversion = tk.StringVar()
        columnas_LSTM_inversion = ('5000', '10000', '15000')
        ttk.Combobox(self.LSTM_frame, values=columnas_LSTM_inversion, width=8, 
                textvariable=self.LSTM_inversion, font=("Arial", 10)).place(x=130, y=85)
        self.LSTM_inversion.set("10000")
        
        
        # Precio previsto
        lb_LSTM_precio_previsto = self.crear_y_posicionar(
            tk.Label,
            master=self.LSTM_frame,
            place_args={'x': 230, 'y': 85},
            text="Pr. Previsto:",
            font=("Arial", 10, "bold"),
            anchor="center", 
            bg= 'dark slate gray', 
            fg="white", 
            width=13
        )  
        
        Tooltip(lb_LSTM_precio_previsto, " Precio previsto por los analistas en los próximos meses." 
        )    
        
        self.LSTM_precio_previsto = tk.StringVar()
        tk.Entry(self.LSTM_frame, textvariable=self.LSTM_precio_previsto, width=10).place(x=350, y=85) 
        try:
            mi_precio_previsto = float(self.info.get("targetMeanPrice"))
            self.LSTM_precio_previsto.set(f"{mi_precio_previsto:.2f}")
        except:
            self.LSTM_precio_previsto.set("N/A") 
            
        # Precio objetivo
        lb_LSTM_precio_objetivo = self.crear_y_posicionar(
            tk.Label,
            master=self.LSTM_frame,
            place_args={'x': 450, 'y': 85},
            text="Pr. Objetivo:",
            font=("Arial", 10, "bold"),
            anchor="center", 
            bg= 'dark slate gray', 
            fg="white",
            width=13
        )  
        
        Tooltip(lb_LSTM_precio_objetivo, " Precio objetivo: \n"
                " Precio al que hay que vender para obtener el beneficio indicado. " 
        )    
        
        self.LSTM_precio_objetivo = tk.StringVar()
        tk.Entry(self.LSTM_frame, textvariable=self.LSTM_precio_objetivo, width=10).place(x=570, y=85) 
        self.LSTM_precio_objetivo.set("") 
            
        # Colocar el boton de calculo
        def LSTM_calculo():
            # Tomamos en valor real
            valor_real = float(self.LSTM_precio.get())
            valor_objetivo = float(self.LSTM_precio_previsto.get())
            com_compra = float(self.LSTM_com_compra.get())
            com_venta = float(self.LSTM_com_venta.get())
            k_inicio = float(self.LSTM_inversion.get())
        
            if k_inicio < valor_real:
                mb.showerror("Advertencia", "El precio de la acción es mayor que la inversión")
                return
        
            n_acciones = round((k_inicio - com_compra) / valor_real,0)
            n_acciones = math.floor((k_inicio - com_compra) / valor_real)
            margen = float(self.LSTM_beneficio.get())
            k_objetivo = (valor_real * n_acciones) + margen + com_venta + com_compra
            precio_objetivo = round(k_objetivo/n_acciones, 2)
            self.LSTM_precio_objetivo.set(precio_objetivo)
            self.LSTM_titulos.set(n_acciones)
    
        style = ttk.Style()        
        style.configure("Ice.TButton",
                foreground="#003366",
                background="#E0F7FA",
                font=("Segoe UI", 10, "italic"),
                padding=6)
        style.map("Ice.TButton",
            background=[("active", "#B2EBF2")],
            foreground=[("active", "#00796B")])
        
          
        btn_inv_LSTM = ttk.Button(self.LSTM_frame, 
                                  text="Calcular Precio", 
                                  style="Ice.TButton",
                                  command=LSTM_calculo,
                                  width=15
                                  ).place(x=570, y=110)
        
        btn_graph_LSTM = ttk.Button(self.LSTM_frame, 
                                    text="Gráfica de test (Real vs Predicción)", 
                                    style="Ice.TButton",
                                    command=self.LSTM_grafica,
                                    width=28
                                    ).place(x=1040, y=10)
        
        btn_graph_LSTM7   = ttk.Button(self.LSTM_frame, 
                                      text="Gráfica de previsión 7 Días", 
                                      style="Ice.TButton",
                                      command=self.LSTM_grafica7,
                                      width=28
                                      ).place(x=1040, y=60)
        
        btn_graph_LSTM30 = ttk.Button(self.LSTM_frame, 
                                      text="Gráfica de previsión 30 Días", 
                                      style="Ice.TButton",
                                      command=self.LSTM_grafica30,
                                      width=28
                                      ).place(x=1040, y=110)
        
        # Beneficio 
        lb_LSTM_beneficio = self.crear_y_posicionar(
            tk.Label,
            master=self.LSTM_frame,
            place_args={'x': 10, 'y': 110},
            text="Bº Inversión:",
            font=("Arial", 10, "bold"),
            anchor="center", 
            bg= 'dark slate gray', 
            fg="white",
            width=13
        )    

        Tooltip(lb_LSTM_beneficio, " Beneficio total de la inversión." 
        )
        
        self.LSTM_beneficio = tk.StringVar()
        columnas_LSTM_beneficio = ('500', '750', '1000')
        ttk.Combobox(self.LSTM_frame, values=columnas_LSTM_beneficio, width=8, 
                textvariable=self.LSTM_beneficio, font=("Arial", 10)).place(x=130, y=110)
        self.LSTM_beneficio.set("500")
        
        # Numero de acciones
        lb_LSTM_titulos = self.crear_y_posicionar(
            tk.Label,
            master=self.LSTM_frame,
            place_args={'x': 230, 'y': 110},
            text="No. Títulos:",
            font=("Arial", 10, "bold"),
            anchor="center", 
            bg= 'dark slate gray', 
            fg="white",
            width=13
        )  
        
        Tooltip(lb_LSTM_titulos, " Número de Títulos a comprar."
        )    
        
        self.LSTM_titulos = tk.StringVar()
        tk.Entry(self.LSTM_frame, textvariable=self.LSTM_titulos, width=10).place(x=350, y=110) 
        self.LSTM_titulos.set("") 
              
    def actualizar_barra(self, valor, n_epo):
        #print('Pasa por la barra')
        self.barra_txt_label.config(text=f"Epoch No: {n_epo} de 100")
        if valor > 100:
            valor = valor % 100  # reinicia cíclicamente

        self.barra["value"] = valor
        self.barra.update()
        self.update_idletasks()
        
    def ver_progreso(self):
        
        self.frame_progreso.place(x=700, y=10) 
        self.barra_txt_label.config(text="")
 
    def ocultar_progreso(self):
        self.frame_progreso.place_forget()    

    def lecturas_entrenamiento(self):
        # Tomar la mejor epoca
         
        # Colocar el tiempo de finalizacion de entreno
        self.tiempo_entreno_final = time.time()
        duracion = self.tiempo_entreno_final - self.tiempo_entreno_inicial
        if duracion <= 60:
            txt = f"Entrenado en: {round(duracion)} seg."
        else:
            minutos = duracion // 60
            segundos = duracion % 60
            txt = f"Entrenado en: {round(minutos)} min. {round(segundos)} seg."
    
        self.lb_tiempo_entreno.config(text=txt)
    
    ####################################
    
    def LSTM_grafica(self):  
        
        # Comprobar que este el precio calculado
        try:
            float(self.LSTM_precio_objetivo.get())
        except:
            mb.showerror("Advertencia", "Calcule el precio objetivo")
            return    
    
        # Inicializamos el tiempo de entrenamiento
        self.tiempo_entreno_inicial = time.time()
        self.lb_tiempo_entreno.config(text='')
        self.ver_progreso()
        
        features = self.datos_completos[['Close','High', 'Low', 'RSI', 'SMA_20', 'EMA_20', 'ATR','VIX','VWAP_D']].dropna()
        
        # 1. Calcular el Logaritmo Natural del Precio de Cierre
        features['Log_Close'] = np.log(features['Close'])

        # 2. Calcular los Retornos Logarítmicos (la diferencia entre Log_Close de hoy y ayer)
        # Este será tu nuevo TARGET (y)
        features['Log_Return'] = features['Log_Close'].diff()
        
        # Eliminar el primer NaN generado por .diff()
        features = features.dropna()

        # 3. Preparación de Features (Ajuste de la Columna TARGET)
        # La columna 'Log_Return' ahora debe ser la columna TARGET para la LSTM.
        # Las features de entrada seguirán siendo las originales, pero nos aseguraremos que 'Log_Return' sea la primera columna.
        # El resto de tus FEATURES (RSI, SMA, EMA) no necesitan esta transformación.

        FEATURES_LOG = ['Log_Return', 'High', 'Low', 'Close', 'RSI', 'SMA_20', 'EMA_20', 'ATR', 'VIX','VWAP_D']
        # NOTA: Incluir 'Close' y 'Log_Return' es intencional, ya que Close es útil como feature de entrada.

        features_log = features[FEATURES_LOG].values
        
        julio_2025 = self.datos_completos.loc['2025-07']
        print('Julio 2025:', julio_2025)
        
        N_STEPS = 40        # Ventana de tiempo (días) de entrada.
        N_SUBSTEPS = 8
        N_TIMESTEPS = N_STEPS // N_SUBSTEPS # 40 / 8 = 5
        N_OUTPUT = 1        # Solo predicción del día siguiente.
        TRAIN_RATIO = 0.80  # 80% para entrenamiento, 20% para validación.
        N_FEATURES = len(FEATURES_LOG)
        
        # Separar target y features
        target = features_log[:, 0].reshape(-1, 1)  # Log_Return
        features = features_log[:, 1:]             # resto de variables

        # Escalar solo las features
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)

        # Escalar el target por separado
        scaler_target = MinMaxScaler()
        target_scaled = scaler_target.fit_transform(target)

        # Unir de nuevo si lo necesitas
        data_scaled = np.hstack([target_scaled, features_scaled])
        
        # B. Función de Secuencias (Single-Step)
        def crear_secuencias(data, n_steps, n_output):
            X, y = [], []
            for i in range(len(data) - n_steps - n_output + 1):
                secuencia = data[i:i + n_steps] 
                # Salida: solo la primera variable (Log_Return) del día siguiente
                objetivo = data[i + n_steps:i + n_steps + n_output, 0] 
                X.append(secuencia)
                y.append(objetivo)
            return np.array(X), np.array(y)

        X_original, y = crear_secuencias(data_scaled, N_STEPS, N_OUTPUT)
        
        # C. RESHAPE PARA CNN-LSTM (Añade la 4ta dimensión)
        X = X_original.reshape((X_original.shape[0], N_SUBSTEPS, N_TIMESTEPS, N_FEATURES))

        # D. División Entrenamiento/Validación (Cronológica)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=(1 - TRAIN_RATIO), shuffle=False
        )
        
        print('X_train:', len(X_train))
        print('X_val:', len(X_val))
        
        print(f"Forma de X_train (CNN-LSTM): {X_train.shape}")
        print(f"Número de features de entrada: {N_FEATURES}")
        
        # --- 4. CONSTRUCCIÓN Y ENTRENAMIENTO DEL MODELO (CNN-LSTM) ---

        model = Sequential()

        # 1. TimeDistributed CNN para extraer patrones locales (en los 5 días)
        model.add(TimeDistributed(
            Conv1D(filters=64, kernel_size=1, activation='relu'), 
            input_shape=(N_SUBSTEPS, N_TIMESTEPS, N_FEATURES)
        ))
        model.add(TimeDistributed(MaxPooling1D(pool_size=1)))
        model.add(TimeDistributed(Dropout(0.1))) # Bajo Dropout
        model.add(TimeDistributed(Flatten())) 
        
        # 2. Capa LSTM para procesar los patrones secuenciales (en los 8 bloques)
        model.add(LSTM(128, activation='relu', return_sequences=False))
        model.add(Dropout(0.1)) 

        # 3. Capa de Salida
        model.add(Dense(N_OUTPUT)) 

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        tkinter_callback = TkinterCallback(self)
        #lista_callbacks = [tkinter_callback]
        
        callbacks = [
            tkinter_callback,
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
        ]

        print("\nEntrenando el modelo CNN-LSTM...")
        history = model.fit(
            X_train, 
            y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1 
        )
        print("Entrenamiento finalizado.")
        
        # --- 5. PREDICCIÓN EN EL CONJUNTO DE PRUEBA (Single-Step) ---

        def reconstruct_prices(start_price, log_returns):
            """Reconstruye precios absolutos a partir de Log-Retornos y un precio inicial."""
            prices = [start_price]
            for r in log_returns:
                next_price = prices[-1] * np.exp(r)
                prices.append(next_price)
            return np.array(prices[1:])

        # 1. Definir el Punto de Partida (Precio 'Ancla')
        # Encuentra el índice del último precio de cierre real (día anterior al inicio de y_val)
        start_index_y = len(self.datos_completos['Close']) - len(X_val) - N_STEPS
        start_index_y = int(X.shape[0] * TRAIN_RATIO)
        
        # Índice del último dato de entrenamiento
        last_train_index = int(X.shape[0] * TRAIN_RATIO) - 1

        
        ultima_fecha = self.datos_completos.index[-1]
        print("La fecha del último registro es:", ultima_fecha)
        print('Start index y:', start_index_y) 
        start_price_idx = self.datos_completos.index[start_index_y]
        initial_anchor_price = self.datos_completos['Close'].loc[:start_price_idx].iloc[-1]

        # 2. Bucle de Predicción y Reconstrucción
        prices_pred_abs = []
        prices_real_abs = []
        current_anchor_price = initial_anchor_price

        # Iterar sobre cada secuencia en X_val
        for i in range(len(X_val)):
            x_input = X_val[i].reshape(1, N_SUBSTEPS, N_TIMESTEPS, N_FEATURES) # Usar la forma CNN-LSTM
            y_real_scaled = y_val[i]

            # Predicción y Desescalado
            y_pred_scaled = model.predict(x_input, verbose=0)[0]
            log_return_pred = scaler_target.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()[0]
            log_return_real = scaler_target.inverse_transform(y_real_scaled.reshape(-1, 1)).flatten()[0]

            # Reconstrucción (basada en el precio real del día anterior para evitar error acumulado)
            pred_price = current_anchor_price * np.exp(log_return_pred)
            real_price = current_anchor_price * np.exp(log_return_real)
            
            prices_pred_abs.append(pred_price)
            prices_real_abs.append(real_price)
            
            # Actualizar el precio ancla
            current_anchor_price = real_price
    

        # 3. Generar Fechas y Gráfico
        start_date_plot = self.datos_completos.index[start_index_y] + pd.Timedelta(days=1)
        print('Start date plot:', start_date_plot)
        plot_dates = pd.date_range(start=start_date_plot, periods=len(X_val), freq='B')
        
        print('Longitud Xval:', len(X_val))
        print('Fechas:', start_date_plot)
        print('Valores:', plot_dates[-10:])
        fechas_objetivo = plot_dates[-10:]
        df_filtrado = self.datos_completos.loc[self.datos_completos.index.isin(fechas_objetivo)]
        print('Precios reales:', prices_real_abs[-10:])
        print('Precios Predichos:', prices_pred_abs[-10:])
        print('Fechas de cierre:', df_filtrado['Close'])
        
        print('Cantidad Precios reales:', len(prices_real_abs))
        print('Cantidad Precios Predichos:', len(prices_pred_abs))
        print('Datos completos:', len(self.datos_completos['Close']))
        
        self.ocultar_progreso()  
        self.lecturas_entrenamiento()   
        
        # Limpiar el canvas anterior si existe

        if hasattr(self, "canvas_LSTM"):
            print('entra en el canvas prob')
            self.canvas_LSTM.get_tk_widget().destroy()
            self.canvas_LSTM = None
            self.fig = None
            self.LSTM_grafico.update_idletasks() # Forzar la actualización del layout del frame
            time.sleep(1)
            
        nmb = self.dame_nombre(self.ticker.get())    
        
        # Tamaño en píxeles
        pixeles_ancho = 1200
        pixeles_alto = 400
        dpi = 100

        # Convertir a pulgadas
        figsize = (pixeles_ancho / dpi, pixeles_alto / dpi)
        
        #fig, (ax, ax1) = plt.subplots(figsize=figsize, dpi=dpi)
        #fig = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Crear una fuente personalizada
        fuente_leyenda = FontProperties(family='Arial', style='italic', size=8)
        
        fig, ax1 = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        
        ax1.plot(plot_dates, prices_real_abs, 
                label='Valor Real (Muestra Test)', 
                color='blue', 
                linewidth=2)

        ax1.plot(plot_dates, prices_pred_abs, 
                label='Predicción CNN-LSTM (Muestra Test)', 
                color='red', 
                linestyle='--', 
                linewidth=2)

        ax1.set_title(f'Acción: {nmb}. Modelo CNN-LSTM - Conjunto de Prueba (Real vs Predicción - A partir de Retornos logarítmicos)')
        # Leyenda con fuente personalizada y ubicación fija
        ax1.legend(loc='upper left', prop=fuente_leyenda)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        ax1.set_xlabel('Fecha')
        ax1.set_ylabel('Escala de Precios')
        ax1.grid(True)
        
        manejador = partial(self.hover, ax1, fig)
        
        fig.canvas.mpl_connect("motion_notify_event", manejador)
        
        
        fig.tight_layout()   
                
        # Insertar gráfico en el Frame
        self.canvas_LSTM = FigureCanvasTkAgg(fig, master=self.LSTM_grafico)
        self.canvas_LSTM.draw()
        self.canvas_LSTM.get_tk_widget().pack(fill='both', expand=True)  
        
    ############################ PROXIMOS 30 DIAS #######################################
        
    def LSTM_grafica30(self):  
        
        # Comprobar que este el precio calculado
        try:
            float(self.LSTM_precio_objetivo.get())
        except:
            mb.showerror("Advertencia", "Calcule el precio objetivo")
            return   
    
        # Inicializamos el tiempo de entrenamiento
        self.tiempo_entreno_inicial = time.time()
        self.lb_tiempo_entreno.config(text='')
        self.ver_progreso()
        
        features = self.datos_completos[['Close','High', 'Low', 'RSI', 'SMA_20', 'EMA_20', 'ATR','VIX','VWAP_D']].dropna()
        
        # 1. Calcular el Logaritmo Natural del Precio de Cierre
        features['Log_Close'] = np.log(features['Close'])

        # 2. Calcular los Retornos Logarítmicos (la diferencia entre Log_Close de hoy y ayer)
        # Este será tu nuevo TARGET (y)
        features['Log_Return'] = features['Log_Close'].diff()
        
        # Eliminar el primer NaN generado por .diff()
        features = features.dropna()

        # 3. Preparación de Features (Ajuste de la Columna TARGET)
        # La columna 'Log_Return' ahora debe ser la columna TARGET para la LSTM.
        # Las features de entrada seguirán siendo las originales, pero nos aseguraremos que 'Log_Return' sea la primera columna.
        # El resto de tus FEATURES (RSI, SMA, EMA) no necesitan esta transformación.

        FEATURES_LOG = ['Log_Return', 'High', 'Low', 'Close', 'RSI', 'SMA_20', 'EMA_20', 'ATR', 'VIX','VWAP_D']
        # NOTA: Incluir 'Close' y 'Log_Return' es intencional, ya que Close es útil como feature de entrada.

        features_log = features[FEATURES_LOG].values
        
        N_STEPS = 40        # Ventana de tiempo (días) de entrada.
        N_SUBSTEPS = 8
        N_TIMESTEPS = N_STEPS // N_SUBSTEPS # 40 / 8 = 5
        N_OUTPUT = 30       # Predicción de los próximos 30 dias.
        TRAIN_RATIO = 0.80  # 80% para entrenamiento, 20% para validación.
        N_FEATURES = len(FEATURES_LOG)
        
        # Separar target y features
        target = features_log[:, 0].reshape(-1, 1)  # Log_Return
        features = features_log[:, 1:]             # resto de variables

        # Escalar solo las features
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)

        # Escalar el target por separado
        scaler_target = MinMaxScaler()
        target_scaled = scaler_target.fit_transform(target)

        # Unir de nuevo si lo necesitas
        data_scaled = np.hstack([target_scaled, features_scaled])
        
        # B. Función de Secuencias (Multi-Step)
        def crear_secuencias(data, n_steps, n_output):
            X, y = [], []
            for i in range(len(data) - n_steps - n_output + 1):
                secuencia = data[i:i + n_steps] 
                # Salida: n_output días de Log_Return (índice 0)
                objetivo = data[i + n_steps:i + n_steps + n_output, 0] 
                X.append(secuencia)
                y.append(objetivo)
            return np.array(X), np.array(y)

        X_original, y = crear_secuencias(data_scaled, N_STEPS, N_OUTPUT)
        
        # C. RESHAPE PARA CNN-LSTM (Añade la 4ta dimensión)
        X = X_original.reshape((X_original.shape[0], N_SUBSTEPS, N_TIMESTEPS, N_FEATURES))

        # D. División Entrenamiento/Validación (Cronológica)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=(1 - TRAIN_RATIO), shuffle=False
        )
        
        print(f"Forma de X_train (CNN-LSTM): {X_train.shape}")
        print(f"Número de features de entrada: {N_FEATURES}")
        
        # --- 4. CONSTRUCCIÓN Y ENTRENAMIENTO DEL MODELO (CNN-LSTM) ---

        model = Sequential()

        # 1. TimeDistributed CNN para extraer patrones locales (en los 5 días)
        model.add(TimeDistributed(
            Conv1D(filters=64, kernel_size=1, activation='relu'), 
            input_shape=(N_SUBSTEPS, N_TIMESTEPS, N_FEATURES)
        ))
        model.add(TimeDistributed(MaxPooling1D(pool_size=1)))
        model.add(TimeDistributed(Dropout(0.1))) # Bajo Dropout
        model.add(TimeDistributed(Flatten())) 
        
        # 2. Capa LSTM para procesar los patrones secuenciales (en los 8 bloques)
        model.add(LSTM(128, activation='relu', return_sequences=False))
        model.add(Dropout(0.1)) 

        # 3. Capa de Salida
        model.add(Dense(N_OUTPUT)) 

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        tkinter_callback = TkinterCallback(self)
        
        callbacks = [
            tkinter_callback,
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
        ]

        print("\nEntrenando el modelo CNN-LSTM...")
        history = model.fit(
            X_train, 
            y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1 
        )
        print("Entrenamiento finalizado.")
        
        # --- 5. PREDICCIÓN DE LOS PRÓXIMOS 30 DÍAS Y VISUALIZACIÓN ---

        def reconstruct_prices(start_price, log_returns):
            """Reconstruye precios absolutos a partir de Log-Retornos y un precio inicial."""
            prices = [start_price]
            for r in log_returns:
                next_price = prices[-1] * np.exp(r)
                prices.append(next_price)
            return np.array(prices[1:])

        # A. Preparar la Secuencia (Últimos 40 días disponibles)
        last_sequence_original = data_scaled[-N_STEPS:] 
        x_input_future = last_sequence_original.reshape(1, N_SUBSTEPS, N_TIMESTEPS, N_FEATURES)

        # B. Predicción de los 30 Log-Retornos futuros
        y_pred_scaled = model.predict(x_input_future, verbose=0)[0]

        # C. Desescalado de la Predicción
        y_pred_log_returns = scaler_target.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

        # D. Reconstrucción del Precio Absoluto (Inversión recursiva)
        last_real_price = self.datos_completos['Close'].iloc[-1] 
        prices_pred_abs = reconstruct_prices(last_real_price, y_pred_log_returns)

        # E. Generación de Fechas y Gráfico
        PREDICTION_DAYS = N_OUTPUT
        future_dates = pd.date_range(
            start=self.datos_completos.index[-1] + pd.Timedelta(days=1), 
            periods=PREDICTION_DAYS, 
            freq='B' # Días hábiles
        )
        prediction_df = pd.DataFrame(prices_pred_abs, index=future_dates, columns=['Predicción'])

        # Historial para contexto
        HISTORICAL_DAYS = 90 
        historical_prices = self.datos_completos['Close'].iloc[-HISTORICAL_DAYS:]


        print("\n--- Resultados de la Predicción a 30 Días ---")
        print(prediction_df)
        
        self.ocultar_progreso()  
        self.lecturas_entrenamiento()   
        
        # Limpiar el canvas anterior si existe

        if hasattr(self, "canvas_LSTM"):
            print('entra en el canvas prob')
            self.canvas_LSTM.get_tk_widget().destroy()
            self.canvas_LSTM = None
            self.fig = None
            self.LSTM_grafico.update_idletasks() # Forzar la actualización del layout del frame
            time.sleep(1)
            
        fuente_leyenda = FontProperties(family='Arial', style='italic', size=8)    
            
        nmb = self.dame_nombre(self.ticker.get())    
        
        # Tamaño en píxeles
        pixeles_ancho = 1200
        pixeles_alto = 400
        dpi = 100

        # Convertir a pulgadas
        figsize = (pixeles_ancho / dpi, pixeles_alto / dpi)
        
        #fig, (ax, ax1) = plt.subplots(figsize=figsize, dpi=dpi)
        #fig = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Crear una fuente personalizada
        fuente_leyenda = FontProperties(family='Arial', style='italic', size=8)
        
        fig, ax1 = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        
        # 1. Historial Real 
        ax1.plot(historical_prices.index, 
                historical_prices.values, 
                label='Histórico Real', 
                color='blue', 
                linewidth=2)

        # 2. Predicción de 30 Días
        ax1.plot(prediction_df.index, 
                prediction_df['Predicción'].values, 
                label='Predicción CNN-LSTM 30 Días', 
                color='red', 
                linestyle='--', 
                linewidth=2)

        # 3. Conexión 
        ax1.plot([historical_prices.index[-1], prediction_df.index[0]], 
                [historical_prices.values[-1], prediction_df['Predicción'].values[0]], 
                color='red', 
                linestyle='--', 
                alpha=0.6)
        
        
        ax1.axhline(float(self.LSTM_precio_objetivo.get()), 
                    color='green', 
                    linestyle='--', 
                    linewidth=1, 
                    label=f'Precio objetivo: {round(float(self.LSTM_precio_objetivo.get()),2)}'
                    )
        
        ax1.set_title(f'Predicción de Precio de la Acción de {nmb} (30 Días Futuros) con CNN-LSTM')
        # Leyenda con fuente personalizada y ubicación fija
        ax1.legend(loc='upper left', prop=fuente_leyenda)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        ax1.set_xlabel('Fecha')
        ax1.set_ylabel('Precio de Cierre (USD)')
        ax1.grid(True)
        
        fig.tight_layout()   
                
        # Insertar gráfico en el Frame
        self.canvas_LSTM = FigureCanvasTkAgg(fig, master=self.LSTM_grafico)
        self.canvas_LSTM.draw()
        self.canvas_LSTM.get_tk_widget().pack(fill='both', expand=True)  
        
    #################################################################################
    ############################### PROXIMOS SIETE DIAS #############################
    #################################################################################    
            
        
    def LSTM_grafica7(self):  
        
        # Comprobar que este el precio calculado
        try:
            float(self.LSTM_precio_objetivo.get())
        except:
            mb.showerror("Advertencia", "Calcule el precio objetivo")
            return   
    
        # Inicializamos el tiempo de entrenamiento
        self.tiempo_entreno_inicial = time.time()
        self.lb_tiempo_entreno.config(text='')
        self.ver_progreso()
        
        features = self.datos_completos[['Close','High', 'Low', 'RSI', 'SMA_20', 'EMA_20', 'ATR','VIX','VWAP_D']].dropna()
        
        # 1. Calcular el Logaritmo Natural del Precio de Cierre
        features['Log_Close'] = np.log(features['Close'])

        # 2. Calcular los Retornos Logarítmicos (la diferencia entre Log_Close de hoy y ayer)
        # Este será tu nuevo TARGET (y)
        features['Log_Return'] = features['Log_Close'].diff()
        
        # Eliminar el primer NaN generado por .diff()
        features = features.dropna()

        # 3. Preparación de Features (Ajuste de la Columna TARGET)
        # La columna 'Log_Return' ahora debe ser la columna TARGET para la LSTM.
        # Las features de entrada seguirán siendo las originales, pero nos aseguraremos que 'Log_Return' sea la primera columna.
        # El resto de tus FEATURES (RSI, SMA, EMA) no necesitan esta transformación.

        FEATURES_LOG = ['Log_Return', 'High', 'Low', 'Close', 'RSI', 'SMA_20', 'EMA_20', 'ATR', 'VIX','VWAP_D']
        # NOTA: Incluir 'Close' y 'Log_Return' es intencional, ya que Close es útil como feature de entrada.

        features_log = features[FEATURES_LOG].values
        
        N_STEPS = 40        # Ventana de tiempo (días) de entrada.
        N_SUBSTEPS = 8
        N_TIMESTEPS = N_STEPS // N_SUBSTEPS # 40 / 8 = 5
        N_OUTPUT = 7        # Predicción de los próximos 7 dias.
        TRAIN_RATIO = 0.80  # 80% para entrenamiento, 20% para validación.
        N_FEATURES = len(FEATURES_LOG)
        
        # Separar target y features
        target = features_log[:, 0].reshape(-1, 1)  # Log_Return
        features = features_log[:, 1:]             # resto de variables

        # Escalar solo las features
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)

        # Escalar el target por separado
        scaler_target = MinMaxScaler()
        target_scaled = scaler_target.fit_transform(target)

        # Unir de nuevo si lo necesitas
        data_scaled = np.hstack([target_scaled, features_scaled])
        
        # B. Función de Secuencias (Multi-Step)
        def crear_secuencias(data, n_steps, n_output):
            X, y = [], []
            for i in range(len(data) - n_steps - n_output + 1):
                secuencia = data[i:i + n_steps] 
                # Salida: n_output días de Log_Return (índice 0)
                objetivo = data[i + n_steps:i + n_steps + n_output, 0] 
                X.append(secuencia)
                y.append(objetivo)
            return np.array(X), np.array(y)

        X_original, y = crear_secuencias(data_scaled, N_STEPS, N_OUTPUT)
        
        # C. RESHAPE PARA CNN-LSTM (Añade la 4ta dimensión)
        X = X_original.reshape((X_original.shape[0], N_SUBSTEPS, N_TIMESTEPS, N_FEATURES))

        # D. División Entrenamiento/Validación (Cronológica)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=(1 - TRAIN_RATIO), shuffle=False
        )
        
        print(f"Forma de X_train (CNN-LSTM): {X_train.shape}")
        print(f"Número de features de entrada: {N_FEATURES}")
        
        # --- 4. CONSTRUCCIÓN Y ENTRENAMIENTO DEL MODELO (CNN-LSTM) ---

        model = Sequential()

        # 1. TimeDistributed CNN para extraer patrones locales (en los 5 días)
        model.add(TimeDistributed(
            Conv1D(filters=64, kernel_size=1, activation='relu'), 
            input_shape=(N_SUBSTEPS, N_TIMESTEPS, N_FEATURES)
        ))
        model.add(TimeDistributed(MaxPooling1D(pool_size=1)))
        model.add(TimeDistributed(Dropout(0.1))) # Bajo Dropout
        model.add(TimeDistributed(Flatten())) 
        
        # 2. Capa LSTM para procesar los patrones secuenciales (en los 8 bloques)
        model.add(LSTM(128, activation='relu', return_sequences=False))
        model.add(Dropout(0.1)) 

        # 3. Capa de Salida
        model.add(Dense(N_OUTPUT)) 

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        tkinter_callback = TkinterCallback(self)
        
        callbacks = [
            tkinter_callback,
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
        ]

        print("\nEntrenando el modelo CNN-LSTM...")
        history = model.fit(
            X_train, 
            y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1 
        )
        print("Entrenamiento finalizado.")
        
        # --- 5. PREDICCIÓN DE LOS PRÓXIMOS 7 DÍAS Y VISUALIZACIÓN ---

        def reconstruct_prices(start_price, log_returns):
            """Reconstruye precios absolutos a partir de Log-Retornos y un precio inicial."""
            prices = [start_price]
            for r in log_returns:
                next_price = prices[-1] * np.exp(r)
                prices.append(next_price)
            return np.array(prices[1:])

        # A. Preparar la Secuencia (Últimos 40 días disponibles)
        last_sequence_original = data_scaled[-N_STEPS:] 
        x_input_future = last_sequence_original.reshape(1, N_SUBSTEPS, N_TIMESTEPS, N_FEATURES)

        # B. Predicción de los 7 Log-Retornos futuros
        y_pred_scaled = model.predict(x_input_future, verbose=0)[0]

        # C. Desescalado de la Predicción
        y_pred_log_returns = scaler_target.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

        # D. Reconstrucción del Precio Absoluto (Inversión recursiva)
        last_real_price = self.datos_completos['Close'].iloc[-1] 
        prices_pred_abs = reconstruct_prices(last_real_price, y_pred_log_returns)

        # E. Generación de Fechas y Gráfico
        PREDICTION_DAYS = N_OUTPUT
        future_dates = pd.date_range(
            start=self.datos_completos.index[-1] + pd.Timedelta(days=1), 
            periods=PREDICTION_DAYS, 
            freq='B' # Días hábiles
        )
        prediction_df = pd.DataFrame(prices_pred_abs, index=future_dates, columns=['Predicción'])

        # Historial para contexto
        HISTORICAL_DAYS = 90 
        historical_prices = self.datos_completos['Close'].iloc[-HISTORICAL_DAYS:]


        print("\n--- Resultados de la Predicción a 30 Días ---")
        print(prediction_df)
        
        self.ocultar_progreso()  
        self.lecturas_entrenamiento()   
        
        # Limpiar el canvas anterior si existe

        if hasattr(self, "canvas_LSTM"):
            print('entra en el canvas prob')
            self.canvas_LSTM.get_tk_widget().destroy()
            self.canvas_LSTM = None
            self.fig = None
            self.LSTM_grafico.update_idletasks() # Forzar la actualización del layout del frame
            time.sleep(1)
            
        fuente_leyenda = FontProperties(family='Arial', style='italic', size=8)    
            
        nmb = self.dame_nombre(self.ticker.get())    
        
        # Tamaño en píxeles
        pixeles_ancho = 1200
        pixeles_alto = 400
        dpi = 100

        # Convertir a pulgadas
        figsize = (pixeles_ancho / dpi, pixeles_alto / dpi)
        
        #fig, (ax, ax1) = plt.subplots(figsize=figsize, dpi=dpi)
        #fig = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Crear una fuente personalizada
        fuente_leyenda = FontProperties(family='Arial', style='italic', size=8)
        
        fig, ax1 = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        
        # 1. Historial Real 
        ax1.plot(historical_prices.index, 
                historical_prices.values, 
                label='Histórico Real', 
                color='blue', 
                linewidth=2)

        # 2. Predicción de 30 Días
        ax1.plot(prediction_df.index, 
                prediction_df['Predicción'].values, 
                label='Predicción CNN-LSTM 7 Días', 
                color='red', 
                linestyle='--', 
                linewidth=2)

        # 3. Conexión 
        ax1.plot([historical_prices.index[-1], prediction_df.index[0]], 
                [historical_prices.values[-1], prediction_df['Predicción'].values[0]], 
                color='red', 
                linestyle='--', 
                alpha=0.6)
        
        
        ax1.axhline(float(self.LSTM_precio_objetivo.get()), 
                    color='green', 
                    linestyle='--', 
                    linewidth=1, 
                    label=f'Precio objetivo: {round(float(self.LSTM_precio_objetivo.get()),2)}'
                    )
        
        ax1.set_title(f'Predicción de Precio de la Acción de {nmb} (7 Días Futuros) con CNN-LSTM')
        # Leyenda con fuente personalizada y ubicación fija
        ax1.legend(loc='upper left', prop=fuente_leyenda)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        ax1.set_xlabel('Fecha')
        ax1.set_ylabel('Precio de Cierre (USD)')
        ax1.grid(True)
        
        
        fig.tight_layout()   
                
        # Insertar gráfico en el Frame
        self.canvas_LSTM = FigureCanvasTkAgg(fig, master=self.LSTM_grafico)
        self.canvas_LSTM.draw()
        self.canvas_LSTM.get_tk_widget().pack(fill='both', expand=True)  
        
    ##########################################
    
    # 3. FUNCIÓN DE MANEJO DE EVENTOS (CALLBACK)
    
        self.annot = ax1.annotate(
            "Coordenadas",
            xy=(0, 0),
            xytext=(20, 20),  # Desplazamiento del texto (offset) respecto al punto (xy)
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w", alpha=0.8),  # Estilo de la caja (fondo blanco semi-transparente)
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color="k"),
            visible=False
        )
    
    
    def hover(self, gr, fig, event):
        """Función que se llama cada vez que el ratón se mueve."""
        
        # Comprueba si el evento ocurrió sobre el área de los ejes
        if event.inaxes == gr:
            # Obtener las coordenadas X e Y del cursor en los datos del gráfico
            coord_x_num = event.xdata
            coord_y = event.ydata
            
            # Coordenadas en píxeles de la pantalla (para la lógica de límites)
            coord_x_pix = event.x
            coord_y_pix = event.y
            
            # Obtener el renderer del canvas
            renderer = fig.canvas.get_renderer()
    
            # Obtener la transformación de datos a píxeles del eje (ax)
            transform = gr.transData
            
            # 1. OBTENER LOS LÍMITES DEL EJE EN PÍXELES
            xlim = gr.get_xlim()
            ylim = gr.get_ylim()
            
            # Convertir los límites del eje (datos) a píxeles
            # [(x_min, y_min), (x_max, y_max)] en píxeles
            min_pix = transform.transform((xlim[0], ylim[0]))
            max_pix = transform.transform((xlim[1], ylim[1]))

            x_min_pix, y_min_pix = min_pix
            x_max_pix, y_max_pix = max_pix

            # Definir el desplazamiento base y el margen de seguridad en píxeles
            offset_x = 15  # Desplazamiento inicial a la derecha
            offset_y = 15  # Desplazamiento inicial arriba
            margen = 100   # Píxeles desde el borde donde debe cambiar el offset
            ha = 'left'   # Horizontal Alignment (Izquierda)
            va = 'bottom' # Vertical Alignment (Abajo)
            
            
            # A. Comprobar el límite DERECHO
            if x_max_pix - coord_x_pix < margen:
                # Mover el texto a la IZQUIERDA del punto.
                print("Activando desplazamiento a la IZQUIERDA")
                offset_x = -15
                ha = 'right' # Anclar el texto a la derecha del punto
            else:
                offset_x = 15
                ha = 'left'  # Anclar el texto a la izquierda del punto
            # B. Comprobar el límite SUPERIOR e INFERIOR
            # Nota: Matplotlib (y muchos backends) tiene Y=0 en la parte INFERIOR
            
            # Comprobar borde SUPERIOR (Y-píxel alto)
            if coord_y_pix > y_max_pix - margen: 
                # Si está cerca del borde superior, mover el texto ABAJO.
                offset_y = -15
                va = 'top' # Anclar el texto en la parte superior del punto
    
            # Comprobar borde INFERIOR (Y-píxel bajo)
            elif coord_y_pix < y_min_pix + margen:
                # Si está cerca del borde inferior, mover el texto ARRIBA.
                offset_y = 15    
                va = 'bottom' # Anclar el texto en la parte inferior del punto            
            else:
                # Para el centro, podemos dejar el offset y centrar verticalmente si es necesario
                va = 'center'        
                    
            # 1. Convertir el número decimal de fecha a objeto datetime
            # Es crucial usar num2date y la zona horaria del eje si se especificó,
            # aunque si no se especificó, la conversión simple funciona.
            try:
                # Convertir a datetime
                fecha_dt = mdates.num2date(coord_x_num)
        
                # Formatear la fecha como cadena de texto (ejemplo: '2023-10-17')
                # Puedes cambiar el formato según necesites: '%Y-%m-%d %H:%M:%S'
                fecha_str = fecha_dt.strftime('%Y-%m-%d') 
            except ValueError:
                # En caso de que el cursor esté en una zona sin datos válidos
                fecha_str = "Fecha no válida"

            # 1. Formatear y actualizar el texto del tooltip
            text = f"Fecha: {fecha_str}\nPrecio: {coord_y:.3f}" # .3f para 3 decimales
            self.annot.set_text(text)

            # 2. Actualizar la posición del tooltip
            # Ajusta la alineación horizontal y vertical
            self.annot.set_ha(ha)
            self.annot.set_va(va)
            
            
            # 3. APLICAR EL NUEVO DESPLAZAMIENTO (OFFSET)
            self.annot.xytext = (offset_x, offset_y) # Ajusta el desplazamiento del texto
            # Esto establece el punto (xy) al que apunta la flecha
            self.annot.xy = (coord_x_num, coord_y)

            # 3. Hacer visible el tooltip
            self.annot.set_visible(True)

            # 4. Redibujar la figura para mostrar los cambios
            # draw_idle() es mejor para eventos de movimiento que el draw() normal
            fig.canvas.draw_idle()
        else:
            # Si el ratón está fuera de los ejes, ocultar el tooltip
            if self.annot.get_visible():
                self.annot.set_visible(False)
                fig.canvas.draw_idle()

    
    ############################################    
        
        


        
        
        
        
        
        
        
        
        
    def cerrarVentana(self):
        self.quit()
        self.destroy()    

def main():
    mi_app = Aplicacion()
    return 0

if __name__ == '__main__':
    main()              
        
        
#pyinstaller --onefile principal.py 
        
        
        
        
        
        
        
        
        
        
                
        
        
        
        
