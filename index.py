import pandas as pd
import pickle 
import streamlit as st 
  
pickle_in = open('hotelbookingclassification.pkl', 'rb') 
classifier = pickle.load(pickle_in) 

dict_label = {}
dict_label['hotel'] = {'City Hotel': 0, 'Resort Hotel': 1}
dict_label['arrival_date_month'] = {'April': 0, 'August': 1, 'December': 2, 'February': 3, 'January': 4, 'July': 5, 'June': 6, 'March': 7, 'May': 8, 'November': 9, 'October': 10, 'September': 11}
dict_label['meal'] = {'BB': 0, 'FB': 1, 'HB': 2, 'SC': 3, 'Undefined': 4}
dict_label['country'] = {'ABW': 0, 'AGO': 1, 'ALB': 2, 'AND': 3, 'ARE': 4, 'ARG': 5, 'ARM': 6, 'ASM': 7, 'ATA': 8, 'ATF': 9, 'AUS': 10, 'AUT': 11, 'AZE': 12, 'BDI': 13, 'BEL': 14, 'BEN': 15, 'BFA': 16, 'BGD': 17, 'BGR': 18, 'BHR': 19, 'BHS': 20, 'BIH': 21, 'BLR': 22, 'BOL': 23, 'BRA': 24, 'BRB': 25, 'CAF': 26, 'CHE': 27, 'CHL': 28, 'CHN': 29, 'CIV': 30, 'CMR': 31, 'CN': 32, 'COL': 33, 'COM': 34, 'CPV': 35, 'CRI': 36, 'CUB': 37, 'CYM': 38, 'CYP': 39, 'CZE': 40, 'DEU': 41, 'DMA': 42, 'DNK': 43, 'DOM': 44, 'DZA': 45, 'ECU': 46, 'EGY': 47, 'ESP': 48, 'EST': 49, 'ETH': 50, 'FIN': 51, 'FRA': 52, 'FRO': 53, 'GAB': 54, 'GBR': 55, 'GEO': 56, 'GGY': 57, 'GHA': 58, 'GIB': 59, 'GLP': 60, 'GNB': 61, 'GRC': 62, 'GTM': 63, 'GUY': 64, 'HKG': 65, 'HND': 66, 'HRV': 67, 'HUN': 68, 'IDN': 69, 'IMN': 70, 'IND': 71, 'IRL': 72, 'IRN': 73, 'IRQ': 74, 'ISL': 75, 'ISR': 76, 'ITA': 77, 'JAM': 78, 'JEY': 79, 'JOR': 80, 'JPN': 81, 'KAZ': 82, 'KEN': 83, 'KHM': 84, 'KIR': 85, 'KNA': 86, 'KOR': 87, 'KWT': 88, 'LAO': 89, 'LBN': 90, 'LBY': 91, 'LCA': 92, 'LIE': 93, 'LKA': 94, 'LTU': 95, 'LUX': 96, 'LVA': 97, 'MAC': 98, 'MAR': 99, 'MCO': 100, 'MDG': 101, 'MDV': 102, 'MEX': 103, 'MKD': 104, 'MLI': 105, 'MLT': 106, 'MMR': 107, 'MNE': 108, 'MOZ': 109, 'MRT': 110, 'MUS': 111, 'MYS': 112, 'MYT': 113, 'NAM': 114, 'NCL': 115, 'NGA': 116, 'NIC': 117, 'NLD': 118, 'NOR': 119, 'NPL': 120, 'NZL': 121, 'OMN': 122, 'PAK': 123, 'PAN': 124, 'PER': 125, 'PHL': 126, 'PLW': 127, 'POL': 128, 'PRI': 129, 'PRT': 130, 'PRY': 131, 'PYF': 132, 'QAT': 133, 'ROU': 134, 'RUS': 135, 'RWA': 136, 'SAU': 137, 'SDN': 138, 'SEN': 139, 'SGP': 140, 'SLE': 141, 'SLV': 142, 'SMR': 143, 'SRB': 144, 'STP': 145, 'SUR': 146, 'SVK': 147, 'SVN': 148, 'SWE': 149, 'SYC': 150, 'SYR': 151, 'TGO': 152, 'THA': 153, 'TJK': 154, 'TMP': 155, 'TUN': 156, 'TUR': 157, 'TWN': 158, 'TZA': 159, 'UGA': 160, 'UKR': 161, 'UMI': 162, 'URY': 163, 'USA': 164, 'UZB': 165, 'VEN': 166, 'VGB': 167, 'VNM': 168, 'ZAF': 169, 'ZMB': 170, 'ZWE': 171}
dict_label['market_segment'] = {'Aviation': 0, 'Complementary': 1, 'Corporate': 2, 'Direct': 3, 'Groups': 4, 'Offline TA/TO': 5, 'Online TA': 6, 'Undefined': 7}
dict_label['distribution_channel'] = {'Corporate': 0, 'Direct': 1, 'GDS': 2, 'TA/TO': 3, 'Undefined': 4}
dict_label['reserved_room_type'] = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'L': 8}
dict_label['assigned_room_type'] = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10}
dict_label['deposit_type'] = {'No Deposit': 0, 'Non Refund': 1, 'Refundable': 2}
dict_label['customer_type'] = {'Contract': 0, 'Group': 1, 'Transient': 2, 'Transient-Party': 3}

  
def prediction(h,lt,ady,adm,adw,add, wdn,wkn,ad,ch,bb,me,cn,ms,dc,rg,pc,pnc,rrt,art,bc,dt,ag,cp,wld,ct,adr,rcp,tsr):   
    li = []
    tmp = {}
    tmp['hotel'] = dict_label['hotel'].get(h)
    tmp['lead_time'] = lt # 0,424
    tmp['arrival_date_year'] = ady # 2015,2017
    tmp['arrival_date_month'] = dict_label['arrival_date_month'].get(adm)
    tmp['arrival_date_week_number'] = adw # 0,53
    tmp['arrival_date_day_of_month'] = add # 0,31
    tmp['stays_in_weekend_nights'] = wdn # 0,3
    tmp['stays_in_week_nights'] = wkn # 0,8
    tmp['adults'] =  ad # 1,10
    tmp['children'] = ch # 1,10
    tmp['babies'] = bb # 1,10
    tmp['meal'] = dict_label['meal'].get(me)
    tmp['country'] = dict_label['country'].get(cn)
    tmp['market_segment'] = dict_label['market_segment'].get(ms)
    tmp['distribution_channel'] = dict_label['distribution_channel'].get(dc)
    tmp['is_repeated_guest'] = rg # 0,1
    tmp['previous_cancellations'] = pc # 1,10
    tmp['previous_bookings_not_canceled'] = pnc # 1,10
    tmp['reserved_room_type'] = dict_label['reserved_room_type'].get(rrt)
    tmp['assigned_room_type'] = dict_label['assigned_room_type'].get(art)
    tmp['booking_changes'] = bc # 1,10
    tmp['deposit_type'] = dict_label['deposit_type'].get(dt)
    tmp['agent'] =  ag # 1,400
    tmp['company'] =  cp # 1,400
    tmp['days_in_waiting_list'] = wld # 0,55
    tmp['customer_type'] = dict_label['customer_type'].get(ct)
    tmp['adr'] = adr # 0.253
    tmp['required_car_parking_spaces'] = rcp
    tmp['total_of_special_requests'] = tsr # 0,2
    print(tmp)
    li.append(tmp)
    s = classifier.predict(pd.DataFrame(li))
    if s == 0:
        return "Genuine"
    else:
        return "Fake"

      
dict_H = set(dict_label['hotel'].keys())
dict_DM = set(dict_label['arrival_date_month'].keys())
dict_ME = set(dict_label['meal'].keys())
dict_CN = set(dict_label['country'].keys())
dict_MS = set(dict_label['market_segment'].keys())
dict_DC = set(dict_label['distribution_channel'].keys()) 
dict_DT = set(dict_label['deposit_type'].keys())
dict_RT = set(dict_label['reserved_room_type'].keys())
dict_AT = set(dict_label['assigned_room_type'].keys())
dict_CT = set(dict_label['customer_type'].keys())
  
def main(): 
    st.header("Hotel Booking Classification (G8:Capstone)")

    col1, col2, col3 = st.columns(3)
    with col1:
        h = st.selectbox("Hotel", dict_H)
        lt = st.number_input("Lead Time", min_value=0, max_value=424)
        ad = st.number_input("Adults", min_value=0, max_value=10)
        ch = st.number_input("Children", min_value=0, max_value=10)
        bb = st.number_input("Babies", min_value=0, max_value=10)
        me = st.selectbox("Meal", dict_ME)
        cn = st.selectbox("Country", dict_CN)
        ms = st.selectbox("Market Segment", dict_MS)
        dc = st.selectbox("Distribution Channel", dict_DC)
    with col2:
        add = st.number_input("Date", min_value=0, max_value=31)
        adm = st.selectbox("Month", dict_DM)
        ady = st.selectbox("Month", (2015,2016,2017))
        adw = st.number_input("Week", min_value=0, max_value=53)
        wdn = st.number_input("Weekend nights", min_value=0, max_value=3)
        wkn = st.number_input("Week nights", min_value=0, max_value=8)
        rg = st.number_input("Repeated Guests", min_value=0, max_value=10)
        pc = st.number_input("Previous Cancellation", min_value=0, max_value=10)
        pnc = st.number_input("Previous not Cancellation", min_value=0, max_value=10)
        

    with col3:
        ag = st.number_input("Agent", min_value=0, max_value=400)
        cp = st.number_input("Company", min_value=0, max_value=400)
        wld = st.number_input("Waiting list days", min_value=0, max_value=55)
        bc = st.number_input("Booking Changes", min_value=0, max_value=10)
        rrt = st.selectbox("Reserved Room", dict_RT)
        art = st.selectbox("Assigned Room", dict_AT)
        ct = st.selectbox("Customer Type", dict_CT)
        dt = st.selectbox("Deposit Type", dict_DT)
        adr = 0
        rcp = 0
        tsr = 0

    result ="" 
    if st.button("Forecast"): 
        result = prediction(h,lt,ady,adm,adw,add, wdn,wkn,ad,ch,bb,me,cn,ms,dc,rg,pc,pnc,rrt,art,bc,dt,ag,cp,wld,ct,adr,rcp,tsr) 
        st.success('This Hotel booking is {}'.format(result))
    else:
        st.success('Fill the form to find about hotel booking.') 
     
if __name__=='__main__': 
    main() 