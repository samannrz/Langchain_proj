def write_to_gsheet(service_file_path, spreadsheet_id, sheet_name, data_df):
    """
    this function takes data_df and writes it under spreadsheet_id
    and sheet_name using your credentials under service_file_path
    """
    gc = pygsheets.authorize(service_file=service_file_path)
    sh = gc.open_by_key(spreadsheet_id)
    try:
        sh.add_worksheet(sheet_name)
    except:
        pass
    wks_write = sh.worksheet_by_title(sheet_name)
    wks_write.clear('A1', None, '*')
    wks_write.set_dataframe(data_df, (1, 1), encoding='utf-8', fit=True)
    wks_write.frozen_rows = 1
####################################
####################################
link_list=[]
title_list=[]
departement_list=[]
import urllib.request
import pygsheets
import pandas as pd

myrange = range(1000,9999)
#myrange = range(1264,1270)
for i in myrange:
    print(i)
    mylink = 'https://www.rdv-prefecture.interieur.gouv.fr/rdvpref/reservation/demarche/'+ str(i)+'/cgu'
    try:
        webUrl=urllib.request.urlopen(mylink)
        htmldata=webUrl.read()
        htmldata = (str(htmldata))
        index1 = htmldata.find('<title')
        index2 = htmldata.find('</title>')
        title = (htmldata[index1+13:index2])
        title_new = title.replace('&quot','\"')
        title_new = title_new.replace('\\xc3\\xa9','e')
        title_new = title_new.replace('\\xc3\\xa8','e')
        title_new = title_new.replace(';','')
        title_new = title_new.replace('/ Constituez votre dossier','')
        title_new = title_new.replace('\\xc3\\xa0','a')
        title_new = title_new.replace('\\xc3\\x89', 'E')
        title_new = title_new.replace('\\xe2\\x82\\xac', 'Euro')
        title_new = title_new.replace('\\xc3\\xa2', 'a')
        title_new = title_new.replace('\\xc3\\xb4', 'o')
        title_new = title_new.replace('\\xc3\\xa7', 'c')



        departement_index = htmldata.find('d\\xc3\\xa9partement')
        if departement_index != -1:
            departement = (htmldata[departement_index+18:departement_index+34])
            departement = departement.replace('&#39;','\'')
            departement = departement.replace('\\xc3\\xa9','e')
            departement = departement.replace(';','')
        else:
            departement = ''

        departement_list.append(departement)
        link_list.append(mylink)
        title_list.append(title_new)
    except:
        pass



data_df = pd.DataFrame(
    {'Link': link_list, 'Title': title_list, 'Departement': departement_list})
#print(data_df)
sfpath = 'keycode/my-gpysheets-3d8d13442005.json'
sheetID = '1czYSf5eYYpvUwESuYh00o4FEjSBkkBuhXIB3Rrph0XI'
sheetName = 'Sheet1'
write_to_gsheet(sfpath, sheetID, sheetName, data_df)
