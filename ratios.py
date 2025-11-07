import numpy as np
from datetime import datetime

class Financials:
    def __init__(self):
        self.Tax_Effect_Of_Unusual_Items = None	                                #Efecto fiscal de partidas inusuales
        self.Tax_Rate_For_Calcs = None	                                        #Tasa impositiva para cálculos
        self.Normalized_EBITDA = None	                                        #EBITDA normalizado
        self.Total_Unusual_Items = None	                                        #Total de partidas inusuales
        self.Total_Unusual_Items_Excluding_Goodwill = None	                    #Total de partidas inusuales excluyendo plusvalía
        self.Net_Income_From_Continuing_Operation_Net_Minority_Interest = None	#Ingreso neto de operaciones continuas neto de interés minoritario
        self.Reconciled_Depreciation = None	                                    #Depreciación conciliada
        self.Reconciled_Cost_Of_Revenue = None	                                #Costo de ingresos conciliado

        self.EBITDA = None	                                                #EBITDA
        self.EBIT = None	                                                #EBIT (Beneficio antes de intereses e impuestos)
        self.Net_Interest_Income = None	                                    #Ingreso neto por intereses
        self.Interest_Expense = None	                                    #Gastos por intereses
        self.Interest_Income = None	                                        #Ingresos por intereses
        self.Normalized_Income = None	                                    #Ingreso normalizado
        self.Net_Income_From_Continuing_And_Discontinued_Operation = None	#Ingreso neto de operaciones continuas y discontinuas
        self.Total_Expenses = None	                                        #Gastos totales
        self.Rent_Expense_Supplemental = None	                            #Gasto suplementario por alquiler
        self.Total_Operating_Income_As_Reported = None	                    #Ingreso operativo total reportado
        self.Diluted_Average_Shares = None	                                #Promedio de acciones diluidas
        self.Basic_Average_Shares = None	                                #Promedio de acciones básicas
        self.Diluted_EPS = None	                                            #Beneficio por acción diluido
        self.Basic_EPS = None	                                            #Beneficio por acción básico
        self.Diluted_NI_Availto_Com_Stockholders = None	                    #Ingreso neto diluido disponible para accionistas comunes
        self.Net_Income_Common_Stockholders = None	                        #Ingreso neto para accionistas comunes
        self.Otherunder_Preferred_Stock_Dividend = None	                    #Otros bajo dividendos de acciones preferentes
        self.Net_Income = None	                                            #Ingreso neto

        self.Minority_Interests = None	                            #Intereses minoritarios
        self.Net_Income_Including_Noncontrolling_Interests = None	#Ingreso neto incluyendo intereses no controladores
        self.Net_Income_Continuous_Operations = None	            #Ingreso neto de operaciones continuas
        self.Tax_Provision = None	                                #Provisión para impuestos
        self.Pretax_Income = None	                                #Ingreso antes de impuestos
        self.Other_Non_Operating_Income_Expenses = None	            #Otros ingresos/gastos no operativos
        self.Special_Income_Charges = None	                        #Cargos especiales de ingresos
        self.Other_Special_Charges = None	                        #Otros cargos especiales
        self.Impairment_Of_Capital_Assets = None	                #Deterioro de activos de capital
        self.Net_Non_Operating_Interest_Income_Expense = None	    #Ingreso/gasto neto por intereses no operativos
        self.Interest_Expense_Non_Operating = None	                #Gastos por intereses no operativos
        self.Interest_Income_Non_Operating = None	                #Ingresos por intereses no operativos
        
        self.Operating_Income = None	                                #Ingreso operativo
        self.Operating_Expense = None	                                #Gasto operativo
        self.Other_Operating_Expenses = None	                        #Otros gastos operativos
        self.Depreciation_And_Amortization_In_Income_Statement = None	#Depreciación y amortización en el estado de resultados
        self.Amortization = None	                                    #Amortización
        self.Depreciation_Income_Statement = None	                    #Depreciación en el estado de resultados
        self.Selling_General_And_Administration = None	                #Gastos de venta, generales y administrativos
        self.Selling_And_Marketing_Expense = None	                    #Gastos de ventas y marketing
        self.General_And_Administrative_Expense = None	                #Gastos generales y administrativos
        self.Rent_And_Landing_Fees = None	                            #Alquiler y tasas de aterrizaje
        self.Gross_Profit = None	                                    #Beneficio bruto
        self.Cost_Of_Revenue = None	                                    #Costo de Oingresos
        self.Total_Revenue = None	                                    #Ingresos totales
        self.Operating_Revenue = None	                                #Ingresos operativos
        
        self.dicc = None
        self.sector = None

    def set_completa_clase(self, dato):
        # Convertir el nombre de los indices con espacio en guiones inferiores
        miDato = dato.copy()
        miDato.index = [i.replace(" ", "_") for i in miDato.index]
        self.dicc = miDato.to_dict()
        #for clave, valor in self.dicc.items():
            #print(f"Clave: {clave} -> Valor: {valor}")
        
        
    def set_genera_partidas(self):
        self.ventas = self.dicc.get("Total_Revenue", 0) 
        self.ventas_operativas = self.dicc.get("Operating_Revenue", 0)
        self.beneficio_neto = self.dicc.get("Net_Income", 0)
        self.coste_ventas = self.dicc.get("Cost_Of_Revenue", 0)
        self.beneficio_bruto = self.dicc.get("Gross_Profit", 0)
        self.beneficio_bruto = self.ventas - self.coste_ventas
        
        self.ingresos_operativos = self.dicc.get("Operating_Income", 0)
        self.gastos_operativos = self.dicc.get("Operating_Expense", 0)
        #self.gastos_operativos = self.beneficio_bruto - self.ingresos_operativos
        self.resultado_operativo = self.ingresos_operativos - self.gastos_operativos
        self.BAII = self.dicc.get("EBIT", 0)
        self.BAII = self.beneficio_bruto - self.gastos_operativos
        
        self.gastos_financieros = self.dicc.get("Interest_Expense", 0)
        
        if self.sector == 'Financial Services':
            # Se trata de un banco
            self.ingresos_financieros = self.dicc.get("Interest_Income", 0)
            self.gastos_financieros = 0
            
        
        self.BAI = self.dicc.get("Pretax_Income", 0)
        self.BAI = self.BAII - self.gastos_financieros
        self.impuestos = self.dicc.get("Tax_Provision", 0)
        self.beneficio_neto = self.BAI - self.impuestos
        
    def set_sector(self, valor):
        self.sector = valor    
        
    def get_ventas(self):
        return self.ventas
        
    def get_coste_ventas(self):
        return self.coste_ventas
        
    def get_ventas_operativas(self):
        return self.ventas_operativas
        
    def get_beneficio_bruto(self):
        return self.beneficio_bruto
          
    def get_resultado_operativo(self):
        return self.beneficio_bruto

    def get_BAII(self):
        return self.BAII

    def get_gastos_financieros(self):
        return self.gastos_financieros

    def get_BAI(self):
        return self.BAI
    
    def get_gastos_operativos(self):
        return self.gastos_operativos
    
    def get_beneficio_neto(self):
        return self.beneficio_neto
    
    def get_impuestos(self):
        return self.impuestos
    
class Balance:
    def __init__(self):

        self.Treasury_Shares_Number = None                             #Número de acciones en autocartera,nan
        self.Ordinary_Shares_Number = None                             #Número de acciones ordinarias,3114746154.0
        self.Share_Issued = None                                       #Acciones emitidas,3114746154.0
        self.Total_Debt = None                                         #Deuda total,5909000000.0
        self.Tangible_Book_Value = None                                #Valor contable tangible,18069000000.0
        self.Invested_Capital = None                                   #Capital invertido,19683000000.0
        self.Working_Capital = None                                    #Capital de trabajo,6169000000.0
        self.Net_Tangible_Assets = None                                #Activos tangibles netos,18069000000.0
        self.Capital_Lease_Obligations = None                          #Obligaciones por arrendamientos financieros,5902000000.0
        self.Common_Stock_Equity = None                                #Patrimonio neto ordinario,19676000000.0
        self.Total_Capitalization = None                               #Capitalización total,19676000000.0
        self.Total_Equity_Gross_Minority_Interest = None               #Patrimonio neto bruto incluyendo minoritarios,19676000000.0
        self.Minority_Interest = None                                  #Intereses minoritarios,0.0
        self.Stockholders_Equity = None                                #Patrimonio neto,19676000000.0
        self.Treasury_Stock = None                                     #Acciones propias en autocartera,47000000.0
        self.Retained_Earnings = None                                  #Resultados acumulados,18994000000.0
        self.Additional_Paid_In_Capital = None                         #Prima de emisión de acciones,20000000.0
        self.Capital_Stock = None                                      #Capital social,94000000.0
        self.Common_Stock = None                                       #Capital social ordinario,94000000.0
        self.Total_Liabilities_Net_Minority_Interest = None            #Pasivo total neto de intereses minoritarios,15038000000.0
        self.Total_Non_Current_Liabilities_Net_Minority_Interest = None   #Pasivo no corriente neto minoritarios,4851000000.0
        self.Other_Non_Current_Liabilities = None                      #Otros pasivos no corrientes,71000000.0
        self.Non_Current_Deferred_Taxes_Liabilities = None             #Pasivos fiscales diferidos no corrientes,72000000.0
        self.Long_Term_Debt_And_Capital_Lease_Obligation = None        #Deuda y arrendamientos financieros a largo plazo,4360000000.0
        self.Long_Term_Capital_Lease_Obligation = None                 #Arrendamientos financieros a largo plazo,4360000000.0
        self.Long_Term_Debt = None                                     #Deuda a largo plazo,0.0
        self.Long_Term_Provisions = None                               #Provisiones a largo plazo,348000000.0
        self.Current_Liabilities = None                                #Pasivo corriente,10187000000.0
        self.Other_Current_Liabilities = None                          #Otros pasivos corrientes,1542138000.0
        self.Current_Debt_And_Capital_Lease_Obligation = None          #Deuda y arrendamientos financieros a corto plazo,1549000000.0
        self.Current_Capital_Lease_Obligation = None                   #Arrendamientos financieros a corto plazo,1542000000.0
        self.Current_Debt = None                                       #Deuda a corto plazo #,7000000.0
        self.Pensionand_Other_Post_Retirement_Benefit_Plans_Current = None   #Planes de pensiones y beneficios post-jubilación,nan
        self.Payables = None                                           #Cuentas por pagar,8590000000.0
        self.Other_Payable = None                                      #Otros pagos pendientes,2289000000.0
        self.Total_Tax_Payable = None                                  #Impuestos por pagar,312000000.0
        self.Accounts_Payable = None                                   #Proveedores,5989000000.0
        self.Total_Assets = None                                       #Activos totales,34714000000.0
        self.Total_Non_Current_Assets = None                           #Activos no corrientes,18358000000.0
        self.Other_Non_Current_Assets = None                           #Otros activos no corrientes,48000000.0
        self.Non_Current_Prepaid_Assets = None                         #Activos prepagados no corrientes,170000000.0
        self.Non_Current_Deferred_Taxes_Assets = None                  #Activos fiscales diferidos no corrientes,800000000.0
        self.Investmentin_Financial_Assets = None                      #Inversiones en activos financieros,26000000.0
        self.Held_To_Maturity_Securities = None                        #Valores mantenidos hasta vencimiento,nan
        self.Available_For_Sale_Securities = None                      #Valores disponibles para la venta,26000000.0
        self.Long_Term_Equity_Investment = None                        #Inversiones en acciones a largo plazo,386000000.0
        self.Investments_In_Other_Ventures_Under_Equity_Method = None      #Inversiones en otras empresas por método de participación,386000000.0
        self.Investment_Properties = None                              #Propiedades de inversión,9000000.0
        self.Goodwill_And_Other_Intangible_Assets = None               #Fondo de comercio y otros activos intangibles,1607000000.0
        self.Other_Intangible_Assets = None                            #Otros activos intangibles,1411000000.0
        self.Goodwill = None                                           #Fondo de comercio,196000000.0
        self.Net_PPE = None                                            #Activo fijo neto,15274000000.0
        self.Accumulated_Depreciation = None                           #Depreciación acumulada,-15361000000.0
        self.Gross_PPE = None                                          #Activo fijo bruto,30635000000.0
        self.Construction_In_Progress = None                           #Construcción en curso,1164000000.0
        self.Other_Properties = None                                   #Otras propiedades,1135000000.0
        self.Machinery_Furniture_Equipment = None                      #Maquinaria, mobiliario y equipo,12924000000.0
        self.Buildings_And_Improvements = None                         #Edificios y mejoras,12681000000.0
        self.Land_And_Improvements = None                              #Terrenos y mejoras,2731000000.0
        self.Properties = None                                         #Propiedades,0.0
        self.Current_Assets = None                                     #Activos corrientes,16356000000.0
        self.Other_Current_Assets = None                               #Otros activos corrientes,94000000.0
        self.Hedging_Assets_Current = None                             #Activos de cobertura corrientes,25000000.0
        self.Assets_Held_For_Sale_Current = None                       #Activos mantenidos para la venta,nan
        self.Inventory = None                                          #Inventario,3321000000.0
        self.Other_Inventories = None                                  #Otros inventarios,nan
        self.Finished_Goods = None                                     #Productos terminados,3000000000.0
        self.Work_In_Process = None                                    #Producción en curso,76000000.0                                                  #
        self.Raw_Materials = None                                      #Materias primas,245000000.0
        self.Other_Receivables = None                                  #Otros créditos por cobrar,342000000.0
        self.Taxes_Receivable = None                                   #Créditos fiscales,326000000.0
        self.Accounts_Receivable = None                                #Clientes,746000000.0
        self.Cash_Cash_Equivalents_And_Short_Term_Investments = None   #Efectivo y equivalentes + inversiones a corto plazo,11502000000.0
        self.Other_Short_Term_Investments = None                       #Otras inversiones a corto plazo,5120000000.0
        self.Cash_And_Cash_Equivalents = None                          #Efectivo y equivalentes,6382000000.0
        self.Cash_Equivalents = None                                   #Equivalentes de efectivo,4881000000.0
        self.Cash_Financial = None                                     #Efectivo financiero,1501000000.0
        
    def set_completa_clase(self, dato):
        # Convertir el nombre de los indices con espacio en guiones inferiores
        miDato = dato.copy()
        miDato.index = [i.replace(" ", "_") for i in miDato.index]
        self.dicc = miDato.to_dict()
        #print("Objeto Balance:\n")  
        #for clave, valor in self.dicc.items():
            #print(f"Clave: {clave} -> Valor: {valor}")       
        
    def get_activo_total(self):
        return self.dicc.get("Total_Assets", 0)    
    
    def get_pasivo_total(self):
        return self.dicc.get("Total_Liabilities_Net_Minority_Interest", 0)   
    
    def get_activo_corriente(self):
        Total_Assets = self.dicc.get("Total_Assets", 0)
        Total_Non_Current_Assets = self.dicc.get("Total_Non_Current_Assets", 0)
        return Total_Assets - Total_Non_Current_Assets      # Activo total - Activo no corriente      
        
    def get_pasivo_corriente(self):
        return self.dicc.get("Current_Liabilities", 0)        
        
    def get_inventario(self):
        return self.dicc.get("Inventory", 0)         

    def get_efectivo(self):
        return self.dicc.get("Cash_And_Cash_Equivalents", 0)       

    def get_patrimonio_neto(self):
        return self.dicc.get("Stockholders_Equity", 0)       


class Quarterly_financials:
    
    def __init__(self):
        self.Tax_Effect_Of_Unusual_Items=None                                   #": "Efecto fiscal de partidas inusuales",
        self.Tax_Rate_For_Calcs=None                                            #": "Tasa impositiva para cálculos",
        self.Normalized_EBITDA=None                                             #": "EBITDA normalizado",
        self.Net_Income_From_Continuing_Operation_Net_Minority_Interest=None    #": "Beneficio neto de operaciones continuas neto de intereses minoritarios",
        self.Reconciled_Depreciation=None                                       #": "Depreciación conciliada",
        self.Reconciled_Cost_Of_Revenue=None                                    #": "Costo de ingresos conciliado",
        self.EBITDA=None                                                        #": "EBITDA",
        self.EBIT=None                                                          #": "EBIT",
        self.Net_Interest_Income=None                                           #": "Ingreso neto por intereses",
        self.Interest_Expense=None                                              #": "Gasto por intereses",
        self.Interest_Income=None                                               #": "Ingreso por intereses",
        self.Normalized_Income=None                                             #": "Ingreso normalizado",
        self.Net_Income_From_Continuing_And_Discontinued_Operations= None       #": "Beneficio neto de operaciones continuas y discontinuas",
        self.Total_Expenses= None                                               #": "Gastos totales",
        self.Total_Operating_Income_As_Reported= None                           #": "Ingreso operativo total según lo reportado",
        self.Diluted_Average_Shares= None                                       #": "Promedio de acciones diluidas",
        self.Basic_Average_Shares= None                                         #": "Promedio de acciones básicas",
        self.Diluted_EPS= None                                                  #": "Beneficio por acción diluido",
        self.Basic_EPS= None                                                    #": "Beneficio por acción básico",
        self.Diluted_NI_Availto_Com_Stockholders= None                          #": "Beneficio neto diluido disponible para accionistas comunes",
        self.Net_Income_Common_Stockholders= None                               #": "Beneficio neto para accionistas comunes",
        self.Otherunder_Preferred_Stock_Dividend= None                          #": "Otros bajo dividendos de acciones preferentes",
        self.Net_Income= None                                                   #": "Beneficio neto",
        self.Minority_Interests= None                                           #": "Intereses minoritarios",
        self.Net_Income_Including_Noncontrolling_Interests = None               #": "Beneficio neto incluyendo intereses no controladores",
        self.Net_Income_Continuous_Operations = None                            #": "Beneficio neto de operaciones continuas",
        self.Tax_Provision = None                                               #": "Provisión fiscal",
        self.Pretax_Income = None                                               #": "Ingreso antes de impuestos",
        self.Net_Non_Operating_Interest_Income_Expense = None                   #": "Ingreso/gasto neto por intereses no operativos",
        self.Interest_Expense_Non_Operating = None                              #": "Gasto por intereses no operativos",
        self.Interest_Income_Non_Operating = None                               #": "Ingreso por intereses no operativos",
        self.Operating_Income = None                                            #": "Ingreso operativo",
        self.Operating_Expense = None                                           #": "Gasto operativo",
        self.Other_Operating_Expenses = None                                    #": "Otros gastos operativos",
        self.Depreciation_And_Amortization_In_Income_Statement = None           #": "Depreciación y amortización en el estado de resultados",
        self.Depreciation_Income_Statement = None                               #": "Depreciación en el estado de resultados",
        self.Gross_Profit = None                                                #": "Beneficio bruto",
        self.Cost_Of_Revenue = None                                             #": "Costo de ingresos",
        self.Total_Revenue = None                                               #": "Ingresos totales",
        self.Operating_Revenue = None                                           #": "Ingresos operativos"
        self.sector = None

    def set_sector(self, valor):
        self.sector = valor   

    def set_completa_clase(self, dato):
        # Convertir el nombre de los indices con espacio en guiones inferiores
        self.df = dato.copy()
        self.df.index = [i.replace(" ", "_") for i in self.df.index]
        #print("Objeto Quarterly tipo pandas:\n", self.df)  

    def get_ventas(self, col):
        if "Total_Revenue" in self.df.index:
            pos = self.df.index.get_loc("Total_Revenue")
        else:
            pos = None  # o cualquier valor por defecto
        return self.df.iloc[pos, col] if pos is not None else 0
    
    def get_coste_ventas(self, col):
        if "Cost_Of_Revenue" in self.df.index:
            pos = self.df.index.get_loc("Cost_Of_Revenue")
        else:
            pos = None  # o cualquier valor por defecto
        return self.df.iloc[pos, col] if pos is not None else 0

    
    def get_ingresos_operativos(self, col):
        if "Operating_Revenue" in self.df.index:
            pos = self.df.index.get_loc("Operating_Revenue")
        else:
            pos = None  # o cualquier valor por defecto
        return self.df.iloc[pos, col] if pos is not None else 0        
        
    
    def get_gastos_operativos(self, col):
        if "Operating_Expense" in self.df.index:
            pos = self.df.index.get_loc("Operating_Expense")
        else:
            pos = None  # o cualquier valor por defecto
        return self.df.iloc[pos, col] if pos is not None else 0        
    
    def get_resultado_operativo(self, col):
        ing = self.get_ingresos_operativos(col)
        gto = self.get_gastos_operativos(col)
        return self.dame_numero(ing) - self.dame_numero(gto)
        
    def get_beneficio_bruto(self, col):
        vta = self.get_ventas(col)
        cte = self.get_coste_ventas(col)
        return self.dame_numero(vta) - self.dame_numero(cte)
    
    def get_BAII(self, col):
        ben = self.get_beneficio_bruto(col)
        gto = self.get_gastos_operativos(col)
        return self.dame_numero(ben) - self.dame_numero(gto)
    
    def get_gastos_financieros(self, col):
        if self.sector == "Financial Services":
            return 0
        else :
            if "Interest_Expense" in self.df.index:
                pos = self.df.index.get_loc("Interest_Expense")
            else:
                pos = None  # o cualquier valor por defecto
            return self.df.iloc[pos, col] if pos is not None else 0        
          
    def get_BAI(self, col):
        rdo = self.get_BAII(col)
        ins = self.get_gastos_financieros(col)
        return self.dame_numero(rdo) - self.dame_numero(ins)
    
    def get_impuestos(self, col):
        if "ax_Provision" in self.df.index:
            pos = self.df.index.get_loc("ax_Provision")
        else:
            pos = None  # o cualquier valor por defecto
        return self.df.iloc[pos, col] if pos is not None else 0        

    def get_beneficio_neto(self, col):
        rdo = self.get_BAI(col)
        imp = self.get_impuestos(col)
        return self.dame_numero(rdo) - self.dame_numero(imp)
    
    def get_nombre_columna(self, col):
        try:
            txt = self.df.columns[col]
            res = txt.strftime('%d-%m-%Y')
        except:
            res = ""    
        return res
          
    def dame_numero(self, num):
        try: 
            num = float(num)
        except:
            num = 0
        return num    
    
