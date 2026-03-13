import pandas as pd

iowa_counties = sorted([
    "Adair","Adams","Allamakee","Appanoose","Audubon","Benton","Black Hawk","Boone",
    "Bremer","Buchanan","Buena Vista","Butler","Calhoun","Carroll","Cass","Cedar",
    "Cerro Gordo","Cherokee","Chickasaw","Clarke","Clay","Clayton","Clinton","Crawford",
    "Dallas","Davis","Decatur","Delaware","Des Moines","Dickinson","Dubuque","Emmet",
    "Fayette","Floyd","Franklin","Fremont","Greene","Grundy","Guthrie","Hamilton",
    "Hancock","Hardin","Harrison","Henry","Howard","Humboldt","Ida","Iowa","Jackson",
    "Jasper","Jefferson","Johnson","Jones","Keokuk","Kossuth","Lee","Linn","Louisa",
    "Lucas","Lyon","Madison","Mahaska","Marion","Marshall","Mills","Mitchell","Monona",
    "Monroe","Montgomery","Muscatine","O'Brien","Osceola","Page","Palo Alto","Plymouth",
    "Pocahontas","Polk","Pottawattamie","Poweshiek","Ringgold","Sac","Scott","Shelby",
    "Sioux","Story","Tama","Taylor","Union","Van Buren","Wapello","Warren","Washington",
    "Wayne","Webster","Winnebago","Winneshiek","Woodbury","Worth","Wright"
])

df = pd.read_csv("corn_yield_iowa_county_2020.csv")

csv_counties = sorted(df["county_name"].astype(str).str.title().unique())

missing = sorted(set(iowa_counties) - set(csv_counties))
extra = sorted(set(csv_counties) - set(iowa_counties))

print("CSV count:", len(csv_counties))
print("Missing counties:", missing)
print("Extra counties:", extra)