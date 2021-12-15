# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 18:02:43 2021

@author: Lenovo
"""
import os
import pandas as pd
import glob
#import PyPDF2
import re

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfparser import PDFParser
from io import StringIO
import io

# PDFを読み込んでPandas DFに保存する関数の開発
def corr(num):
    if num<=12467:
        num -= 105
    elif num<=12475:
        num -= 106
    elif num<=12477:
        num -= 107
        
    elif num == 12547:
        num = 12540
    elif num<=12592:
        num -= 108
        
    if num<12353 or num>12541:
        num = 63
    return num


def extract(t2):
    pattern = '\(cid:[0-9]+\)'
    iter = re.finditer(pattern,t2,flags=re.DOTALL)
    s = ''
    e0 = 0
    for match in iter:
        start,end = (int(match.start()),int(match.end()))
        s += t2[e0:start]
        num = int(t2[start+5:end-1])
        num = corr(num)
        s += chr(num)
        e0 = end
    s = re.sub('[\n]+','\n',s)
    s = re.sub('[" "]+',' ',s)
    return s
    
def convert_pdf_to_txt(path, txtname=None, buf=True, bytesObj=False):

    rsrcmgr = PDFResourceManager()
    if buf:
        outfp = StringIO()

    else:
        outfp = open(txtname, 'w')

    codec = 'utf-8'
    laparams = LAParams()
    laparams.detect_vertical = True
    device = TextConverter(rsrcmgr, outfp, laparams=laparams)
    if bytesObj:
        fp = path
    else:
        fp = open(path, 'rb')

    interpreter = PDFPageInterpreter(rsrcmgr, device)
    for page in PDFPage.get_pages(fp):
        interpreter.process_page(page)

    fp.close()
    device.close()

    if buf:
        #text = re.sub(space, "", outfp.getvalue())
        text = extract(outfp.getvalue())

    outfp.close()
    return text

def pypdf(file):
    from pypdf2 import PyPDF2
    with open(file,'rb') as pdffile:
        pdfreader = PyPDF2.PdfFileReader(pdffile)
        pages = pdfreader.numPages
        doc = ''
        for i in range(pages):
            pageObj = pdfreader.getPage(i)
            doc += pageObj.extractText()
    return doc

base = 'C:/Users/Lenovo/Downloads/pdf_unreadable/'
s = convert_pdf_to_txt(os.path.join(base,'ekx/ekwagon_manual_sales_1905.pdf'))


def separate():
    l1 = ['D3/17_D3_1803.pdf', 'D3/20_lineup_vol_1_2002.pdf', 'D3/catalog_d3_17_1712.pdf', 'D3/catalog_d3_17acc_1703.pdf', 'D3/m_selection_YS3023_1912.pdf', 'D3/mitsubishi_motors_collection_200219.pdf', 'D5/18-5_D5_1910.pdf', 'D5/d5_acc_1911.pdf', 'D5/leaf_d5_4wd_190917.pdf', 'D5/m_selection_YS3023_1912.pdf', 'D5/mitsubishi_motors_collection_200219.pdf', 'PHEV/190905_3_phev.pdf', 'PHEV/19_phev_Acc_1810.pdf', 'PHEV/20_PHEV_2003.pdf', 'PHEV/20_PHEV_BE_2005.pdf', 'PHEV/20_lineup_vol_1_2002.pdf', 'PHEV/20_phev_Acc_1260.pdf', 'PHEV/PHEV_A3tool_LAST.pdf', 'PHEV/PHEV_YS3018_210170.pdf', 'PHEV/acc_value_outlander.pdf', 'PHEV/all_blacks_series_3SXYBAB190.pdf', 'PHEV/all_blacks_series_3SXYBAB190_price.pdf', 'PHEV/all_blacks_series_3SXYBAB201.pdf', 'PHEV/all_blacks_series_3SXYBAB201_ver2.pdf', 'PHEV/charging_equipment_202103.pdf', 'PHEV/charging_leaf_1003_final.pdf', 'PHEV/dynamic_pricing.pdf', 'PHEV/leaf_phev_4wd_1812.pdf', 'PHEV/leaf_phev_4wd_190917.pdf', 'PHEV/m_selection_YS3023_1912.pdf', 'PHEV/mitsubishi_motors_collection_200219.pdf', 'PHEV/mmc_cev_b.pdf', 'PHEV/phev_leaf_disaster_1901.pdf', 'PHEV/phev_sm_0928_link.pdf', 'PHEV/value_phev.pdf', 'PHEV/value_phev_ver2.pdf', 'RVR/leaf_rvr_4wd_190917.pdf', 'RVR/value_rvr_ver2.pdf', 'ekX_âXâyü[âX/202002_staff_manual_QA.pdf', 'ekX_âXâyü[âX/value_ekxspace_ekspace_ver2.pdf', 'ekx/190401_ekx_hosho.pdf', 'ekx/190425_ekx_hikaku_days.pdf', 'ekx/190613_ekx_wagon_hikaku.pdf', 'ekx/20_eK_X_T_Plus_Edition_2003.pdf', 'ekx/ekwagon_manual_sales_1905.pdf', 'ekx/leaf_eclipse_4wd_190917_no_link.pdf', 'ekx/value_ekx_ekwagon_ver2.pdf', 'ekâXâyü[âX/200107_staff_manual_02.pdf', 'ekâÅâSâô/190425_ekx_hikaku_days.pdf', 'ekâÅâSâô/190613_ekx_wagon_hikaku.pdf', 'ekâÅâSâô/ekwagon_manual_sales_1905.pdf', 'ekâÅâSâô/leaf_eclipse_4wd_190917_no_link.pdf', 'ekâÅâSâô/value_ekx_ekwagon_ver2.pdf', 'mirage/20_MIRAGE_1910.pdf', 'mirage/21_mirage_staff_manuel.pdf', 'mirage/mirage_acc_200270.pdf', 'mirage/value_mirage.pdf', 'outlander/19_OUTLANDER_Acc_1903.pdf', 'outlander/19_OUTLANDER_be_1911.pdf', 'outlander/20_OUTLANDER_2003.pdf', 'outlander/20_OUTLANDER_Acc_2003.pdf', 'outlander/leaf_outlander_4wd_190917.pdf', 'outlander/m_selection_YS3023_1912.pdf', 'outlander/mitsubishi_motors_collection_200219.pdf', 'ÉVî^D5/20_D5_Acc_2011.pdf', 'ÉVî^D5/20_D5_HR_1912.pdf', 'ÉVî^D5/20_lineup_vol_1_2002.pdf', 'ÉVî^D5/21_d5_jasper_2012.pdf', 'ÉVî^D5/all_blacks_series_3SXYBAB201.pdf', 'ÉVî^D5/all_blacks_series_3SXYBAB201_ver2.pdf', 'ÉVî^D5/leaf_newd5_4wd_1904.pdf', 'ÉVî^D5/leaf_newd5_4wd_190917.pdf', 'ÉVî^D5/m_selection_YS3023_1912.pdf', 'ÉVî^D5/mitsubishi_motors_collection_200219.pdf', 'ÉVî^D5/value_new_d5_ver2.pdf', 'âAâCâ~ü[âu/18_i-MiEV_1910.pdf', 'âAâCâ~ü[âu/iMiEV_acc_1910.pdf', 'âAâCâ~ü[âu/m_selection_YS3023_1912.pdf', 'âAâCâ~ü[âu/mitsubishi_motors_collection_200219.pdf', 'âGâNâèâvâXâNâìâX/20_ECLIPSECROSS_ACC_2008.pdf', 'âGâNâèâvâXâNâìâX/20_lineup_vol_1_2002.pdf', 'âGâNâèâvâXâNâìâX/22_nse_manual.pdf', 'âGâNâèâvâXâNâìâX/EVremote_QRtag_teisei.pdf', 'âGâNâèâvâXâNâìâX/all_blacks_series_3SXYBAB201_ver2.pdf', 'âGâNâèâvâXâNâìâX/eclipse_accessory_value.pdf', 'âGâNâèâvâXâNâìâX/m_selection_YS3023_1912.pdf', 'âGâNâèâvâXâNâìâX/machiene_plan.pdf', 'âGâNâèâvâXâNâìâX/mitsubishi_motors_collection_200219.pdf', 'âGâNâèâvâXâNâìâX/value_eclipse_ver2.pdf', 'âXâ^âbâtÉΩùp/0_sogo_menu.pdf', 'âXâ^âbâtÉΩùp/18_D5_Acc_1907.pdf', 'âXâ^âbâtÉΩùp/1902_ipad_mquick.pdf', 'âXâ^âbâtÉΩùp/190404_mdf_access2019.pdf', 'âXâ^âbâtÉΩùp/190425_ekx_hikaku_days.pdf', 'âXâ^âbâtÉΩùp/190613_ekx_wagon_hikaku.pdf', 'âXâ^âbâtÉΩùp/1_smp_credit.pdf', 'âXâ^âbâtÉΩùp/1_smp_menu.pdf', 'âXâ^âbâtÉΩùp/200107_staff_manual.pdf', 'âXâ^âbâtÉΩùp/200107_staff_manual_02.pdf', 'âXâ^âbâtÉΩùp/200911_MDF_manual.pdf', 'âXâ^âbâtÉΩùp/20190425_supermycarplan.pdf', 'âXâ^âbâtÉΩùp/20190607_manual_mdf_renkei3.pdf', 'âXâ^âbâtÉΩùp/20190607_manual_mdf_renkei3_2.pdf', 'âXâ^âbâtÉΩùp/20190709_supermycarplan.pdf', 'âXâ^âbâtÉΩùp/20190711_supermycarplan.pdf', 'âXâ^âbâtÉΩùp/20190902_supermycarplan.pdf', 'âXâ^âbâtÉΩùp/20190902_supermycarplan_2.pdf', 'âXâ^âbâtÉΩùp/202002_staff_manual_QA.pdf', 'âXâ^âbâtÉΩùp/202003_hoshu_manual.pdf', 'âXâ^âbâtÉΩùp/202003_shuri_hoshu_manual.pdf', 'âXâ^âbâtÉΩùp/2_smp_return.pdf', 'âXâ^âbâtÉΩùp/3_smp_sales.pdf', 'âXâ^âbâtÉΩùp/4_smp_account_transfer.pdf', 'âXâ^âbâtÉΩùp/5_smp_customer_change.pdf', 'âXâ^âbâtÉΩùp/6_smp_faq.pdf', 'âXâ^âbâtÉΩùp/ACCESS2020_200330.pdf', 'âXâ^âbâtÉΩùp/ACCESS2021_210329.pdf', 'âXâ^âbâtÉΩùp/MDF_manual_210315.pdf', 'âXâ^âbâtÉΩùp/chiku_channel.pdf', 'âXâ^âbâtÉΩùp/cs_smileproject_004.pdf', 'âXâ^âbâtÉΩùp/cs_smileproject_005.pdf', 'âXâ^âbâtÉΩùp/cs_smileproject_guide.pdf', 'âXâ^âbâtÉΩùp/customer_guidance01.pdf', 'âXâ^âbâtÉΩùp/ekwagon_manual_sales_1905.pdf', 'âXâ^âbâtÉΩùp/handotai_list_1.pdf', 'âXâ^âbâtÉΩùp/issue_compact_044.pdf', 'âXâ^âbâtÉΩùp/issue_d5_18.pdf', 'âXâ^âbâtÉΩùp/issue_d5_19.pdf', 'âXâ^âbâtÉΩùp/issue_d5_20.pdf', 'âXâ^âbâtÉΩùp/issue_d5_21.pdf', 'âXâ^âbâtÉΩùp/issue_d5_22.pdf', 'âXâ^âbâtÉΩùp/issue_d5_23.pdf', 'âXâ^âbâtÉΩùp/issue_d5_24.pdf', 'âXâ^âbâtÉΩùp/issue_d5_25.pdf', 'âXâ^âbâtÉΩùp/issue_d5_26.pdf', 'âXâ^âbâtÉΩùp/issue_d5_27.pdf', 'âXâ^âbâtÉΩùp/issue_d5_28.pdf', 'âXâ^âbâtÉΩùp/issue_d5_29.pdf', 'âXâ^âbâtÉΩùp/issue_d5_30.pdf', 'âXâ^âbâtÉΩùp/issue_d5_31.pdf', 'âXâ^âbâtÉΩùp/issue_d5_32.pdf', 'âXâ^âbâtÉΩùp/issue_d5_33.pdf', 'âXâ^âbâtÉΩùp/issue_d5_34.pdf', 'âXâ^âbâtÉΩùp/issue_d5_36.pdf', 'âXâ^âbâtÉΩùp/issue_d5_37.pdf', 'âXâ^âbâtÉΩùp/issue_d5_38.pdf', 'âXâ^âbâtÉΩùp/issue_d5_39.pdf', 'âXâ^âbâtÉΩùp/issue_d5_41.pdf', 'âXâ^âbâtÉΩùp/issue_d5_43.pdf', 'âXâ^âbâtÉΩùp/issue_eclipse_009.pdf', 'âXâ^âbâtÉΩùp/issue_eclipse_010.pdf', 'âXâ^âbâtÉΩùp/issue_eclipse_011.pdf', 'âXâ^âbâtÉΩùp/issue_eclipse_012.pdf', 'âXâ^âbâtÉΩùp/issue_ek_066.pdf', 'âXâ^âbâtÉΩùp/issue_ek_067.pdf', 'âXâ^âbâtÉΩùp/issue_ek_068.pdf', 'âXâ^âbâtÉΩùp/issue_ek_069.pdf', 'âXâ^âbâtÉΩùp/issue_ek_070.pdf', 'âXâ^âbâtÉΩùp/issue_ek_071.pdf', 'âXâ^âbâtÉΩùp/issue_ek_072.pdf', 'âXâ^âbâtÉΩùp/issue_mdf_sp06.pdf', 'âXâ^âbâtÉΩùp/issue_mdf_sp06_2.pdf', 'âXâ^âbâtÉΩùp/issue_phev_085.pdf', 'âXâ^âbâtÉΩùp/issue_phev_086.pdf', 'âXâ^âbâtÉΩùp/issue_phev_087.pdf', 'âXâ^âbâtÉΩùp/issue_phev_088.pdf', 'âXâ^âbâtÉΩùp/issue_phev_089.pdf', 'âXâ^âbâtÉΩùp/issue_phev_90.pdf', 'âXâ^âbâtÉΩùp/issue_phev_91.pdf', 'âXâ^âbâtÉΩùp/issue_phev_92.pdf', 'âXâ^âbâtÉΩùp/issue_phev_93.pdf', 'âXâ^âbâtÉΩùp/issue_phev_94.pdf', 'âXâ^âbâtÉΩùp/issue_rvr_01.pdf', 'âXâ^âbâtÉΩùp/issue_smile_056.pdf', 'âXâ^âbâtÉΩùp/issue_smile_058.pdf', 'âXâ^âbâtÉΩùp/issue_smile_059.pdf', 'âXâ^âbâtÉΩùp/issue_smile_060.pdf', 'âXâ^âbâtÉΩùp/issue_smile_061.pdf', 'âXâ^âbâtÉΩùp/issue_smile_062.pdf', 'âXâ^âbâtÉΩùp/issue_smile_063.pdf', 'âXâ^âbâtÉΩùp/issue_smile_064.pdf', 'âXâ^âbâtÉΩùp/issue_smile_065.pdf', 'âXâ^âbâtÉΩùp/issue_smile_066.pdf', 'âXâ^âbâtÉΩùp/issue_smile_067.pdf', 'âXâ^âbâtÉΩùp/issue_smile_068.pdf', 'âXâ^âbâtÉΩùp/issue_smile_069.pdf', 'âXâ^âbâtÉΩùp/issue_smile_070.pdf', 'âXâ^âbâtÉΩùp/issue_smile_071.pdf', 'âXâ^âbâtÉΩùp/issue_smile_sp1.pdf', 'âXâ^âbâtÉΩùp/issue_smile_sp2.pdf', 'âXâ^âbâtÉΩùp/message_ceo.pdf', 'âXâ^âbâtÉΩùp/mitsubishi_motors_tv.pdf', 'âXâ^âbâtÉΩùp/mmc_youtube.pdf', 'âXâ^âbâtÉΩùp/new_ek_order_list2019.pdf', 'âXâ^âbâtÉΩùp/newd5_staff_manual_1212.pdf', 'âXâ^âbâtÉΩùp/official_youtube.pdf', 'âXâ^âbâtÉΩùp/ultra_employee_sales_2012-1.pdf', 'âXâ^âbâtÉΩùp/ultra_employee_sales_2012-2.pdf', 'âXâ^âbâtÉΩùp/ultra_employee_sales_2104-1.pdf', 'âXâ^âbâtÉΩùp/ultra_m-quick_Manual_2009.pdf', 'âXâ^âbâtÉΩùp/ultra_m-quick_Manual_2104-1.pdf', 'â~âjâLâââuâgâëâbâN/22_minicabtruck_2108.pdf', 'â~âjâLâââuâgâëâbâN/22_minicabtruck_acc_2108.pdf', 'â~âjâLâââuâoâô/20_minicab_van_2104.pdf', 'â~âjâLâââuâoâô/21_minicab_van_Acc_2102.pdf', 'â~âjâLâââuâoâô/22_minicab_van_2109.pdf', 'â~âjâLâââuâoâô/m_selection_YS3023_1912.pdf', 'â~âjâLâââuüEâ~ü[âu/20_MINICABMiEVVAN_2104.pdf', 'â~âjâLâââuüEâ~ü[âu/20_minicab_MiEV_Acc_2009.pdf', 'âëâôâTü[âJü[âSü[/17_Lancer_cargo_1806.pdf', 'âëâôâTü[âJü[âSü[/catalog_lancercargo_17acc_1702.pdf', 'âëâôâTü[âJü[âSü[/m_selection_YS3023_1912.pdf', 'âëâôâTü[âJü[âSü[/mitsubishi_motors_collection_200219.pdf']
    l2 = ['PHEV/21_OUTLANDER_PHEV_2104.pdf', 'PHEV/21_OUTLANDER_PHEV_BE_2104.pdf', 'PHEV/22__outlander_2111.pdf', 'PHEV/new_outlander_manual.pdf', 'RVR/21_RVR_BE_2104.pdf', 'PHEV/19_PHEV_1906.pdf', 'ekâXâyü[âX/22_ekspace_2111.pdf', 'mirage/21_mirage_Acc_2102.pdf', 'ekâXâyü[âX/21_eK-space_HR_2009.pdf', 'ekâÅâSâô/190325_ekx_staff_manual.pdf', 'ekâÅâSâô/21_ek_acc_2110.pdf', 'ekâÅâSâô/22_ek_acc_2111.pdf', 'ÉVî^D5/20_D5_1911.pdf', 'ÉVî^D5/20_D5_2003.pdf', 'ÉVî^D5/20_D5URBANGEAR_2009.pdf', 'ÉVî^D5/21_d5_2012.pdf', 'âGâNâèâvâXâNâìâX/eneos_denki.pdf', 'âXâ^âbâtÉΩùp/190701hojyokin.pdf', 'âXâ^âbâtÉΩùp/issue_d5_42.pdf', 'âXâ^âbâtÉΩùp/manual_ek_mipilot_190313.pdf', 'ekâÅâSâô/190325_manual_ek_mipilot.pdf', 'ekâÅâSâô/22_eK_wagon_2111.pdf', 'PHEV/19_PHEV_G_Ltd_1903.pdf', 'PHEV/19_PHEV_G_Ltd_1906.pdf', 'mirage/21_MIRAGE_BE_2104.pdf', 'outlander/19_OUTLANDER_BE_1902.pdf', 'âGâNâèâvâXâNâìâX/21_ECLIPSECROSS_2104.pdf', 'ekX_âXâyü[âX/21_eK-space_HR_2009.pdf', 'âXâ^âbâtÉΩùp/newd5_staff_manual_1902.pdf', 'âXâ^âbâtÉΩùp/ultra_camp_package.pdf', 'ekX_âXâyü[âX/22_ekX_space_2111.pdf', 'ekX_âXâyü[âX/22_ekx_space_plus_edition_2111.pdf', 'âXâ^âbâtÉΩùp/issue_hojokin_1906.pdf', 'ÉVî^D5/21_D5_2107.pdf', 'ÉVî^D5/21_d5_acc_2012.pdf', 'ÉVî^D5/21_d5_acc_2110.pdf', 'ÉVî^D5/21_D5_JASPER_2107.pdf', 'ÉVî^D5/21_DELICA_D5_2104.pdf', 'ÉVî^D5/21_DELICA_D5_JASPER_2104.pdf', 'ekx/190325_ekx_staff_manual.pdf', 'ekx/22_ek_acc_2111.pdf', 'mirage/21_MIRAGE_2107.pdf', 'PHEV/20_PHEV_BE_2010.pdf', 'PHEV/22_outlander_2110.pdf', 'PHEV/22_outlander_2110_acc.pdf', 'âXâ^âbâtÉΩùp/190325_ekx_staff_manual_2.pdf', 'âXâ^âbâtÉΩùp/190325_manual_ek_mipilot.pdf', 'âGâNâèâvâXâNâìâX/21_eclipse_acc_2011.pdf', 'RVR/22_RVR_ACC_2109.pdf', 'âGâNâèâvâXâNâìâX/20_ECLIPSE_BE_2007.pdf', 'âXâ^âbâtÉΩùp/cs_smileproject_003.pdf', 'ÉVî^D5/20_D5_jasper_2004.pdf', 'ÉVî^D5/20_D5_UG_2003.pdf', 'ÉVî^D5/newd5_acc_1910.pdf', 'âGâNâèâvâXâNâìâX/20_ECLIPSE_CROSS_2007.pdf', 'âGâNâèâvâXâNâìâX/21_eclipse_2011.pdf', 'âGâNâèâvâXâNâìâX/22_eclipse_cross_acc_2111.pdf', 'âGâNâèâvâXâNâìâX/eclipse_pre-order.pdf', 'âXâ^âbâtÉΩùp/190703hojyokin.pdf', 'âXâ^âbâtÉΩùp/191223_staff_manual.pdf', 'ekX_âXâyü[âX/22_ekspace_acc_2111.pdf', 'D2/19_D2_1910.pdf', 'D2/delica_d2_accessory_2.pdf', 'ekx/22_ek_x_g_plus_edition_2111.pdf', 'âXâ^âbâtÉΩùp/issue_d5_40.pdf', 'âXâ^âbâtÉΩùp/issue_phev_084.pdf', 'PHEV/connect_guidebook.pdf', 'PHEV/new_outlander_manual_1112.pdf', 'PHEV/outlander_accessory.pdf', 'RVR/22_RVR_2107.pdf', 'ÉVî^D5/20_D5_2009.pdf', 'ÉVî^D5/20_D5_jasper_2008.pdf', 'ÉVî^D5/21_d5_2012a.pdf', 'ÉVî^D5/21_DELICA_D5_ACC_2103.pdf', 'ÉVî^D5/21_DELICA_D5_ACC_2106.pdf', 'ekx/190325_manual_ek_mipilot.pdf', 'ekx/21_ek_acc_2110.pdf', 'PHEV/21_OUTLANDER_PHEV_ACC_2103.pdf', 'PHEV/22_outlander_acc.pdf', 'ekX_âXâyü[âX/21_eksp_acc_2110.pdf', 'ekx/22_ek_x_2122.pdf', 'D2/DELICA_D2_2107.pdf', 'âGâNâèâvâXâNâìâX/22_eclipse_cross_2111.pdf', 'âXâ^âbâtÉΩùp/190325_ekx_staff_manual.pdf', 'âXâ^âbâtÉΩùp/190704hojyokin.pdf', 'âXâ^âbâtÉΩùp/191210_staff_manual.pdf', 'PHEV/ev_handbook_201124.pdf', 'PHEV/outlander_manual_1028.pdf', 'PHEV/20_PHEV_2010.pdf', 'âXâ^âbâtÉΩùp/manual_ek_190313.pdf', 'ÉVî^D5/20_D5_Acc_2003.pdf', 'ÉVî^D5/21_d5_heartyrun_2012.pdf', 'ÉVî^D5/21_DELICA_D5_3SXTBD5216.pdf', 'ÉVî^D5/delica_d5_2107.pdf', 'âXâ^âbâtÉΩùp/issue_d5_35.pdf']

    src = 'C:/Users/Lenovo/Downloads/MMC SS Navi PDF一覧/'
    d1 = 'C:/Users/Lenovo/Downloads/pdf_unreadable/'
    d2 = 'C:/Users/Lenovo/Downloads/pdf_readable/'
    for path,subdir,files in os.walk(base):
        
        folder = ''
        if len(files)>0:
            folder = path.split('\\')[-1]

        for file in files:
            try:
                if ('l1' if folder+'/'+file in l1 else 'l2') == 'l1':
                    if not os.path.exists(d1+folder):
                        os.mkdir(d1+folder)
                    os.rename(src+folder+'/'+file,d1+folder+'/'+file)
                else:
                    if not os.path.exists(d2+folder):
                        os.mkdir(d2+folder)
                    os.rename(src+folder+'/'+file,d2+folder+'/'+file)
            except Exception as e:
                pass

