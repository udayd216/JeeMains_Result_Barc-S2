import cv2
import typing
import oracledb
import numpy as np
from csv import writer
from selenium import webdriver
from selenium.webdriver.common.by import By
from mltu.configs import BaseModelConfigs
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer

v_PROCESS_USER = 'U1'

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

        # Get model input details
        self.input_name = self.model.get_inputs()[0].name  # Extract input name
        self.input_shape = self.model.get_inputs()[0].shape  # Extract input shape

    def predict(self, image: np.ndarray):
        # Ensure correct width-height order for OpenCV resizing
        image = cv2.resize(image, (self.input_shape[2], self.input_shape[1]))  
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        preds = self.model.run(None, {self.input_name: image_pred})[0]  # Use extracted input_name
        text = ctc_decoder(preds, self.char_list)[0]
        return text

def Data_Xpath_Elements():
    BArch_CRL = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[5]/td/table/tbody/tr[4]/td[1]').text
    BArch_GENEWS = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[5]/td/table/tbody/tr[4]/td[2]').text
    BArch_OBCNCL = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[5]/td/table/tbody/tr[4]/td[3]').text
    BArch_SC = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[5]/td/table/tbody/tr[4]/td[4]').text
    BArch_ST = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[5]/td/table/tbody/tr[4]/td[5]').text
    BArch_CRL_PWBD = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[5]/td/table/tbody/tr[4]/td[6]').text
    BArch_GENEWS_PWBD = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[5]/td/table/tbody/tr[4]/td[7]').text
    BArch_OBCNCL_PWBD = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[5]/td/table/tbody/tr[4]/td[8]').text
    BArch_SC_PWBD = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[5]/td/table/tbody/tr[4]/td[9]').text
    BArch_ST_PWBD = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[5]/td/table/tbody/tr[4]/td[10]').text

    BPlanning_CRL = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[5]/td/table/tbody/tr[5]/td[1]').text
    BPlanning_GENEWS = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[5]/td/table/tbody/tr[5]/td[2]').text
    BPlanning_OBCNCL = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[5]/td/table/tbody/tr[5]/td[3]').text
    BPlanning_SC = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[5]/td/table/tbody/tr[5]/td[4]').text
    BPlanning_ST = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[5]/td/table/tbody/tr[5]/td[5]').text
    BPlanning_CRL_PWBD = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[5]/td/table/tbody/tr[5]/td[6]').text
    BPlanning_GENEWS_PWBD = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[5]/td/table/tbody/tr[5]/td[7]').text
    BPlanning_OBCNCL_PWBD = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[5]/td/table/tbody/tr[5]/td[8]').text
    BPlanning_SC_PWBD = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[5]/td/table/tbody/tr[5]/td[9]').text
    BPlanning_ST_PWBD = driver.find_element(By.XPATH, '//*[@id="tableToPrint"]/div[2]/div/div/table/tbody/tr[5]/td/table/tbody/tr[5]/td[10]').text

    update_IpStatus = ("UPDATE O_JEEMAINS_RES_BARCH_BPLAN_SESS2_25 SET BARCH_CRL = '" + str(BArch_CRL) + "', BARCH_GENEWS = '" + str(BArch_GENEWS) + "', \
        BARCH_OBCNCL = '" + str(BArch_OBCNCL) + "', BARCH_SC = '" + str(BArch_SC) + "', BARCH_ST = '" + str(BArch_ST) + "', \
        BARCH_CRL_PWBD = '" + str(BArch_CRL_PWBD) + "', BARCH_GENEWS_PWBD = '" + str(BArch_GENEWS_PWBD) + "', \
        BARCH_OBCNCL_PWBD = '" + str(BArch_OBCNCL_PWBD) + "', BARCH_SC_PWBD = '" + str(BArch_SC_PWBD) + "', BARCH_ST_PWBD = '" + str(BArch_ST_PWBD) + "', \
        BPLANNING_CRL = '" + str(BPlanning_CRL) + "', BPLANNING_GENEWS = '" + str(BPlanning_GENEWS) + "', \
        BPLANNING_OBCNCL = '" + str(BPlanning_OBCNCL) + "', BPLANNING_SC = '" + str(BPlanning_SC) + "', BPLANNING_ST = '" + str(BPlanning_ST) + "', \
        BPLANNING_CRL_PWBD = '" + str(BPlanning_CRL_PWBD) + "', BPLANNING_GENEWS_PWBD = '" + str(BPlanning_GENEWS_PWBD) + "', \
        BPLANNING_OBCNCL_PWBD = '" + str(BPlanning_OBCNCL_PWBD) + "', BPLANNING_SC_PWBD = '" + str(BPlanning_SC_PWBD) + "', BPLANNING_ST_PWBD = '" + str(BPlanning_ST_PWBD) + "' \
        WHERE APPNO = '"+ str(v_appno) +"'")
    cur.execute(update_IpStatus) # Execute an UPDATE statement
    conn.commit()

    update_IpStatus = "UPDATE I_JEEMAIN_RESULT_SESS2_25 SET PROCESS_STATUS_BARC_BPLAN = 'DONE', PROCESS_USER_BARC_BPLAN = '"+ str(v_PROCESS_USER) +"', CREATEDDATE = SYSDATE WHERE APPNO = '"+ str(v_appno) +"' AND PASSWORD = '"+ str(v_password) +"'"
    cur.execute(update_IpStatus) # Execute an UPDATE statement
    conn.commit()
    
max_attempts = 20
attempts = 0
login_successful = False
    
#oracledb.init_oracle_client()
oracledb.init_oracle_client(lib_dir=r"D:\app\udaykumard\product\instantclient_23_6")
conn = oracledb.connect(user='RESULT', password='LOCALDEV', dsn='192.168.15.196:1521/orcldev')
cur = conn.cursor()

driver = webdriver.Chrome()
driver.maximize_window()

str_dataslot = "SELECT PROCESS_USER, START_VAL, END_VAL FROM DATASLOTS_VAL_USER WHERE PROCESS_USER = '"+v_PROCESS_USER+"'"
cur.execute(str_dataslot)
res_dataslot = cur.fetchall()

start_sno = res_dataslot[0][1]
end_sno = res_dataslot[0][2]

Sno = start_sno

# str_Jeeappno = "SELECT SNO, TRIM(APPNO) APPNO, TRIM(PASSWORD) PASSWORD, ADMNO FROM I_JEEMAIN_RESULT_SESS2_25 \
#             WHERE APPNO = 250310121589"
str_Jeeappno = "SELECT SNO, TRIM(APPNO) APPNO, TRIM(PASSWORD) PASSWORD, ADMNO FROM I_JEEMAIN_RESULT_SESS2_25 \
            WHERE PROCESS_STATUS_BARC_BPLAN = 'D' AND PASSWORD IS NOT NULL AND LENGTH(APPNO) = 12 AND \
            SNO >= '"+str(start_sno)+"' AND  SNO <='"+str(end_sno)+"' ORDER BY SNO"
cur.execute(str_Jeeappno)
res = cur.fetchall()

for row in res:
    v_appno = row[1]     
    v_password = row[2]
    v_AdmNo = row[3]
            
    try:  
        driver.get("https://examinationservices.nic.in/resultservices/JEEMAIN2025S2P2/Login")
        
        i_regno = driver.find_element(By.ID, "txtAppNo")      
        i_regno.send_keys(v_appno)

        i_password = driver.find_element(By.ID, "txtPassword")
        i_password.send_keys(v_password)

        Captchaimg = driver.find_element(By.ID, "capimage")
        driver.execute_script("arguments[0].scrollIntoView(true);", Captchaimg)
        Captchaimg.screenshot('Screenshotcaptcha.png')

        configs = BaseModelConfigs.load("Models/02_captcha_to_text/202502191616/configs.yaml")
        model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

        image = cv2.imread('Screenshotcaptcha.png')
        prediction_text = model.predict(image)

        i_Captcha = driver.find_element(By.ID, "Captcha1")      
        i_Captcha.send_keys(prediction_text)
        
        #Button press
        Submit_button = driver.find_element(By.ID, 'Submit')
        driver.execute_script("arguments[0].click();", Submit_button)
            
        try:
            v_Captcha = driver.find_element(By.XPATH, '/html/body/form/div[2]/div[2]/div/fieldset/div/div/div[6]/div[2]/span').text                  
        except:
            v_Captcha = ''

        try:
            v_invalid = driver.find_element(By.XPATH, '/html/body/form/div[2]/div[2]/div/fieldset/div/div/div[6]/div[2]/span').text 
        except:
            v_invalid = ''
                          
        if v_Captcha == "Invalid CAPTCHA":
            while attempts < max_attempts and not login_successful:
                Captchaimg = driver.find_element(By.ID, "capimage")
                driver.execute_script("arguments[0].scrollIntoView(true);", Captchaimg)
                Captchaimg.screenshot('Screenshotcaptcha.png')

                configs = BaseModelConfigs.load("Models/02_captcha_to_text/202502191616/configs.yaml")
                model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

                image = cv2.imread('Screenshotcaptcha.png')
                prediction_text = model.predict(image)
                
                i_Captcha = driver.find_element(By.ID, "Captcha1")  
                i_Captcha.clear()    
                i_Captcha.send_keys(prediction_text)
                
                #Button press
                Submit_button = driver.find_element(By.ID, 'Submit')
                driver.execute_script("arguments[0].click();", Submit_button)

                try:
                    Data_Xpath_Elements()
                    login_successful = True  
                except:
                    try:
                        v_invalid = driver.find_element(By.XPATH, '/html/body/form/div[2]/div[2]/div/fieldset/div/div/div[6]/div[2]/span').text 
                    except:
                        v_invalid = ''

                    if v_invalid == "Invalid Application Number/Password":        
                        update_IpStatus = "UPDATE I_JEEMAIN_RESULT_SESS2_25 SET PROCESS_STATUS_BARC_BPLAN = 'Invalid', PROCESS_USER_BARC_BPLAN = '"+ str(v_PROCESS_USER) +"', CREATEDDATE = SYSDATE, ERROR_MESSAGE = '"+ str(v_invalid) +"' WHERE APPNO = '"+ str(v_appno) +"' AND PASSWORD = '"+ str(v_password) +"'"
                        cur.execute(update_IpStatus) # Execute an UPDATE statement
                        conn.commit()
                        login_successful=True
                    else:
                        login_successful=False
                        attempts += 1

        elif v_invalid == "Invalid Application Number/Password":        
            update_IpStatus = "UPDATE I_JEEMAIN_RESULT_SESS2_25 SET PROCESS_STATUS_BARC_BPLAN = 'Invalid', PROCESS_USER_BARC_BPLAN = '"+ str(v_PROCESS_USER) +"', CREATEDDATE = SYSDATE, ERROR_MESSAGE = '"+ str(v_invalid) +"' WHERE APPNO = '"+ str(v_appno) +"' AND PASSWORD = '"+ str(v_password) +"'"
            cur.execute(update_IpStatus) # Execute an UPDATE statement
            conn.commit()
        else:
            Data_Xpath_Elements()
            login_successful = True
    except:
        pass

driver.quit()
                                    


