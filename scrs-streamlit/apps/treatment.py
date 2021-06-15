import numpy as np
import pandas as pd

def treatment(diagnose):
    if diagnose.lower() == 'melonoma':
        drug_name = ["Dacarbazine (DTIC)","Temozolomide"]
        treat = ["Dacarbazine is given into a vein. Preferrably, advised by doctor.",
                "This medication is given in a capsule form by mouth. Capsules come in 5 mg, 20 mg, 100 mg, 250 mg, sizes."
                ]
        drugs = {
        "drug_name":drug_name, "treatment":treat
        }
        df = pd.DataFrame(drugs)
        return df


    elif diagnose.lower() == 'Actinic':
        drug_name = ["5-fluorouracil (5-FU) cream", "Diclofenac sodium gel", "Imiquimod cream"]
        treat = ["You apply this once or twice a day for 2 to 4 weeks.",
                "This medication tends to cause less of a skin reaction than 5-FU, but it can still be very effective. You will need to apply it twice a day for 2 to 3 months.",
                "This can be a good option for the face because you can apply it once (or twice) a week, so you don’t get lots of redness and crusting. You may need to apply it for 12 to 16 weeks."]
        drugs = {
        "drug_name":drug_name, "treatment":treat
        }
        df = pd.DataFrame(drugs)
        return df

    if diagnose.lower() == 'basal cell':
        drug_name = ["Cryotherapy", "Laser Surgery"]
        treat = ["Applies liquid nitrogen to the tumor, freezing the abnormal tissue.",
                "Laser surgery only kills tumor cells on the surface of the skin and doesn’t go deeper.",
                ]
        drugs = {
        "drug_name":drug_name, "treatment":treat
        }
        df = pd.DataFrame(drugs)
        print(df)
        return df
