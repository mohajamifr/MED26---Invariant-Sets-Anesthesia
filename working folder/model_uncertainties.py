import numpy as np
from scipy.stats import truncnorm

def lognormal_bounds(p_nom, w, t):
    return p_nom * np.exp(-t * w), p_nom * np.exp(t * w)

def interval_div(a_lo, a_hi, b_lo, b_hi):
    return a_lo / b_hi, a_hi / b_lo

def build_model(v1, v1_lo, v1_hi, v1_real, v2, v2_lo, v2_hi, v2_real, v3, v3_lo, v3_hi, v3_real,
                cl1, cl1_lo, cl1_hi, cl1_real, cl2, cl2_lo, cl2_hi, cl2_real, cl3, cl3_lo, cl3_hi, cl3_real,
                ke0, ke0_lo, ke0_hi, ke0_real):
    
    # variability to ignore
    v1_lo = v1
    v1_hi = v1
    v1_real = v1
    cl1_lo = cl1
    cl1_hi = cl1
    cl1_real = cl1
    ke0_lo = ke0
    ke0_hi = ke0
    ke0_real = ke0

    # rates [1/min]
    k10 = cl1 / v1
    k12 = cl2 / v1
    k13 = cl3 / v1
    k21 = cl2 / v2
    k31 = cl3 / v3

    # rate bounds [1/min]
    k10_lo, k10_hi = interval_div(cl1_lo, cl1_hi, v1_lo, v1_hi)
    k12_lo, k12_hi = interval_div(cl2_lo, cl2_hi, v1_lo, v1_hi)
    k13_lo, k13_hi = interval_div(cl3_lo, cl3_hi, v1_lo, v1_hi)
    k21_lo, k21_hi = interval_div(cl2_lo, cl2_hi, v2_lo, v2_hi)
    k31_lo, k31_hi = interval_div(cl3_lo, cl3_hi, v3_lo, v3_hi)

    # real rates
    k10_real = cl1_real / v1_real
    k12_real = cl2_real / v1_real
    k13_real = cl3_real / v1_real
    k21_real = cl2_real / v2_real
    k31_real = cl3_real / v3_real

    k11 = k10 + k12 + k13
    k11_lo = k10_lo + k12_lo + k13_lo
    k11_hi = k10_hi + k12_hi + k13_hi
    k11_real = k10_real + k12_real + k13_real

    A = np.array([[-k11, k12, k13, 0.0],
                  [k21, -k21, 0.0, 0.0],
                  [k31, 0.0, -k31, 0.0],
                  [ke0, 0.0, 0.0, -ke0]]) / 60.0  # [1/s]

    A_lo = np.array([[-k11_lo, k12_lo, k13_lo, 0.0],
                  [k21_lo, -k21_lo, 0.0, 0.0],
                  [k31_lo, 0.0, -k31_lo, 0.0],
                  [ke0_lo, 0.0, 0.0, -ke0_lo]]) / 60.0  # [1/s]
    
    A_hi = np.array([[-k11_hi, k12_hi, k13_hi, 0.0],
                  [k21_hi, -k21_hi, 0.0, 0.0],
                  [k31_hi, 0.0, -k31_hi, 0.0],
                  [ke0_hi, 0.0, 0.0, -ke0_hi]]) / 60.0  # [1/s]
    
    A_real = np.array([[-k11_real, k12_real, k13_real, 0.0],
                       [k21_real, -k21_real, 0.0, 0.0], 
                       [k31_real, 0.0, -k31_real, 0.0], 
                       [ke0_real, 0.0, 0.0, -ke0_real]]) / 60.0 # [1/s]
    
    B = np.array([[1.0 / v1, 0.0, 0.0, 0.0]]).T  # [1/L]
    B_lo = np.array([[1.0 / v1_hi, 0.0, 0.0, 0.0]]).T  # [1/L]
    B_hi = np.array([[1.0 / v1_lo, 0.0, 0.0, 0.0]]).T  # [1/L]
    B_real = np.array([[1.0 / v1_real, 0.0, 0.0, 0.0]]).T # [1/L]

    return A, A_lo, A_hi, A_real, B, B_lo, B_hi, B_real

def load_model(patient_char=[35, 170, 70, 1], drug='Propofol',
               model='Eleveld', opiate=True, truncated=2.0):
    
    age, height, weight, sex = patient_char
    drug = drug.lower()
    model = model.lower()

    if sex == 1:
        lbm = 1.1 * weight - 128 * (weight/height)**2
    else:
        lbm = 1.07 * weight - 148 * (weight/height)**2

    if drug == "propofol":
        if model == "schnider":

            cl1 = 1.89 + 0.0456 * (weight - 77) - 0.0681 * (lbm - 59) + 0.0264 * (height - 177)
            cl2 = 1.29 - 0.024 * (age - 53)
            cl3 = 0.836

            v1 = 4.27
            v2 = 18.9 - 0.391 * (age - 53)
            v3 = 238

            ke0 = 0.456

            # define ke0 using Eleveld PD model
            # ke0 = 0.146 * (weight/70)**(-0.25)

            # variability
            cv_v1 = v1 * 0.0404
            cv_v2 = v2 * 0.01
            cv_v3 = v3 * 0.1435
            cv_cl1 = cl1 * 0.1005
            cv_cl2 = cl2 * 0.01
            cv_cl3 = cl3 * 0.1179
            cv_ke = ke0 * 0.42

            w_v1 = np.sqrt(np.log(1 + cv_v1**2))
            w_v2 = np.sqrt(np.log(1 + cv_v2**2))
            w_v3 = np.sqrt(np.log(1 + cv_v3**2))
            w_cl1 = np.sqrt(np.log(1 + cv_cl1**2))
            w_cl2 = np.sqrt(np.log(1 + cv_cl2**2))
            w_cl3 = np.sqrt(np.log(1 + cv_cl3**2))
            w_ke0 = np.sqrt(np.log(1 + cv_ke**2))
            # w_ke0 = np.sqrt(0.702)

        elif model == "eleveld":
            AGE_ref = 35
            WGT_ref = 70
            HGT_ref = 1.7
            PMA_ref = (40 + AGE_ref*52) / 52
            BMI_ref = WGT_ref / HGT_ref**2
            GDR_ref = 1

            theta = [None,
                    6.2830780766822, 25.5013145036879, 272.8166615043603,
                    1.7895836588902, 1.83, 1.1085424008536,
                    0.191307, 42.2760190602615, 9.0548452392807,
                    -0.015633, -0.00285709, 33.5531248778544,
                    -0.0138166, 68.2767978846832, 2.1002218877899,
                    1.3042680471360, 1.4189043652084, 0.6805003109141]

            def faging(x): return np.exp(x * (age - AGE_ref))
            def fsig(x, C50, gam): return x**gam / (C50**gam + x**gam)
            def fcentral(x): return fsig(x, theta[12], 1)

            def fal_sallami(sexX, weightX, ageX, bmiX):
                if sexX:
                    return (0.88 + (1-0.88)/(1+(ageX/13.4)**(-12.7))) * (9270*weightX)/(6680+216*bmiX)
                else:
                    return (1.11 + (1-1.11)/(1+(ageX/7.1)**(-1.1))) * (9270*weightX)/(8780+244*bmiX)

            PMA = age + 40/52
            BMI = weight / (height/100)**2

            fCLmat = fsig(PMA * 52, theta[8], theta[9])
            fCLmat_ref = fsig(PMA_ref*52, theta[8], theta[9])
            fQ3mat = fsig(PMA * 52, theta[14], 1)
            fQ3mat_ref = fsig(PMA_ref * 52, theta[14], 1)
            fsal = fal_sallami(sex, weight, age, BMI)
            fsal_ref = fal_sallami(GDR_ref, WGT_ref, AGE_ref, BMI_ref)

            if opiate:
                def fopiate(x): return np.exp(x * age)
            else:
                def fopiate(x): return 1.0

            v1 = theta[1] * fcentral(weight) / fcentral(WGT_ref)
            v2 = theta[2] * weight / WGT_ref * faging(theta[10])
            v2ref = theta[2]
            v3 = theta[3] * fsal / fsal_ref * fopiate(theta[13])
            v3ref = theta[3]

            cl1 = (sex*theta[4] + (1-sex)*theta[15]) * (weight/WGT_ref)**0.75 * fCLmat/fCLmat_ref * fopiate(theta[11])
            cl2 = theta[5] * (v2/v2ref)**0.75 * (1 + theta[16] * (1 - fQ3mat))
            cl3 = theta[6] * (v3/v3ref)**0.75 * fQ3mat/fQ3mat_ref

            ke0 = 0.146 * (weight/70)**(-0.25)

            # log normal standard deviation (as in your snippet)
            w_v1 = np.sqrt(0.610)
            w_v2 = np.sqrt(0.565)
            w_v3 = np.sqrt(0.597)
            w_cl1 = np.sqrt(0.265)
            w_cl2 = np.sqrt(0.346)
            w_cl3 = np.sqrt(0.209)
            w_ke0 = np.sqrt(0.702)

        else:
            raise ValueError("model must be 'schnider' or 'eleveld'.")
        
    elif drug == "remifentanil":
        if model == "minto":

            cl1 = 2.6 - 0.0162 * (age - 40) + 0.0191 * (lbm - 55)
            cl2 = 2.05 - 0.0301 * (age - 40)
            cl3 = 0.076 - 0.00113 * (age - 40)

            v1 = 5.1 - 0.0201 * (age-40) + 0.072 * (lbm - 55)
            v2 = 9.82 - 0.0811 * (age-40) + 0.108 * (lbm-55)
            v3 = 5.42

            ke0 = 0.595 - 0.007 * (age - 40)

            # variability
            cv_v1 = 0.26
            cv_v2 = 0.29
            cv_v3 = 0.66
            cv_cl1 = 0.14
            cv_cl2 = 0.36
            cv_cl3 = 0.41
            cv_ke = 0.68

            w_v1 = np.sqrt(np.log(1 + cv_v1**2))
            w_v2 = np.sqrt(np.log(1 + cv_v2**2))
            w_v3 = np.sqrt(np.log(1 + cv_v3**2))
            w_cl1 = np.sqrt(np.log(1 + cv_cl1**2))
            w_cl2 = np.sqrt(np.log(1 + cv_cl2**2))
            w_cl3 = np.sqrt(np.log(1 + cv_cl3**2))
            w_ke0 = np.sqrt(np.log(1 + cv_ke**2))

        elif model == "eleveld":
            AGE_ref = 35
            WGT_ref = 70
            HGT_ref = 1.7
            BMI_ref = WGT_ref / HGT_ref**2
            GDR_ref = 1

            theta = [None, 
                        2.88,
                        -0.00554,
                        -0.00327,
                        -0.0315,
                        0.470,
                        -0.0260]

            def faging(x): return np.exp(x * (age - AGE_ref))
            def fsig(x, C50, gam): return x**gam / (C50**gam + x**gam)

            def fal_sallami(sexX, weightX, ageX, bmiX):
                if sexX:
                    return (0.88 + (1-0.88)/(1+(ageX/13.4)**(-12.7))) * (9270*weightX)/(6680+216*bmiX)
                else:
                    return (1.11 + (1-1.11)/(1+(ageX/7.1)**(-1.1))) * (9270*weightX)/(8780+244*bmiX)

            BMI = weight / (height/100)**2
            SIZE = (fal_sallami(sex, weight, age, BMI)/fal_sallami(GDR_ref, WGT_ref, AGE_ref, BMI_ref))

            KMAT = fsig(weight, theta[1], 2)
            KMATref = fsig(WGT_ref, theta[1], 2)
            if sex:
                KSEX = 1
            else:
                KSEX = 1+theta[5]*fsig(age, 12, 6)*(1-fsig(age, 45, 6))

            v1ref = 5.81
            v1 = v1ref * SIZE * faging(theta[2])
            V2ref = 8.882
            v2 = V2ref * SIZE * faging(theta[3])
            V3ref = 5.03
            v3 = V3ref * SIZE * faging(theta[4])*np.exp(theta[6]*(weight - WGT_ref))
            cl1ref = 2.58
            cl2ref = 1.72
            cl3ref = 0.124
            cl1 = cl1ref * SIZE**0.75 * (KMAT/KMATref)*KSEX*faging(theta[3])
            cl2 = cl2ref * (v2/V2ref)**0.75 * faging(theta[2]) * KSEX
            cl3 = cl3ref * (v3/V3ref)**0.75 * faging(theta[2])

            ke0 = 1.09 * faging(-0.0289)

            # log normal standard deviation
            w_v1 = np.sqrt(0.104)
            w_v2 = np.sqrt(0.115)
            w_v3 = np.sqrt(0.810)
            w_cl1 = np.sqrt(0.0197)
            w_cl2 = np.sqrt(0.0547)
            w_cl3 = np.sqrt(0.285)
            w_ke0 = np.sqrt(1.26)

        else:
            raise ValueError("model must be 'minto' or 'eleveld'.")
        
    # parameter bounds -> A,B bounds
    t = float(truncated)

    v1_lo, v1_hi = lognormal_bounds(v1, w_v1, t)
    v2_lo, v2_hi = lognormal_bounds(v2, w_v2, t)
    v3_lo, v3_hi = lognormal_bounds(v3, w_v3, t)
    cl1_lo, cl1_hi = lognormal_bounds(cl1, w_cl1, t)
    cl2_lo, cl2_hi = lognormal_bounds(cl2, w_cl2, t)
    cl3_lo, cl3_hi = lognormal_bounds(cl3, w_cl3, t)
    ke0_lo, ke0_hi = lognormal_bounds(ke0, w_ke0, t)

    # real bounds
    v1_real = v1 * np.exp(truncnorm.rvs(-t, t, scale=w_v1))
    v2_real = v2 * np.exp(truncnorm.rvs(-t, t, scale=w_v2))
    v3_real = v3 * np.exp(truncnorm.rvs(-t, t, scale=w_v3))
    cl1_real = cl1 * np.exp(truncnorm.rvs(-t, t, scale=w_cl1))
    cl2_real = cl2 * np.exp(truncnorm.rvs(-t, t, scale=w_cl2))
    cl3_real = cl3 * np.exp(truncnorm.rvs(-t, t, scale=w_cl3))
    ke0_real = ke0 * np.exp(truncnorm.rvs(-t, t, scale=w_ke0))

    # build the model and bounds
    A, A_lo, A_hi, A_real, B, B_lo, B_hi, B_real = build_model(v1, v1_lo, v1_hi, v1_real, v2, v2_lo, v2_hi, v2_real, v3, v3_lo, v3_hi, v3_real,
                                                            cl1, cl1_lo, cl1_hi, cl1_real, cl2, cl2_lo, cl2_hi, cl2_real, cl3, cl3_lo, cl3_hi, cl3_real,
                                                            ke0, ke0_lo, ke0_hi, ke0_real)
    
    return A, A_lo, A_hi, A_real, B, B_lo, B_hi, B_real