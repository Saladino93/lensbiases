"""
Getting lensing reponses.
"""
import numpy as np




#model = "SC"
models = ["TR"]#, "SC", "GM"]#, "SC", "GM"]#, "SC"]#, "SC", "GM"]
#index = models.index(model)

results_n32 = {}


for index, model in enumerate(models):

    print("Working on model", model, ", index", index)

    keys = ["B", "PB"]
    NTOT = {k: [] for k in keys}

    for LL in Ls:
        @vegas.batchintegrand
        def integrand(x):
            l1, l2, theta1, theta2 = x.T
            cos1, cos2 = np.cos(theta1), np.cos(theta2)
            sin1, sin2 = np.sin(theta1), np.sin(theta2)

            l1v = np.array([l1*cos1, l1*sin1])
            l2v = np.array([l2*cos2, l2*sin2])

            l5v = l1v-l2v

            L = np.ones_like(l1)*LL
            Lv = np.c_[L, np.zeros_like(l1)].T

            l4v = Lv-l2v
            
            l5 = np.sqrt((l1*cos1-l2*cos2)**2+(l1*sin1-l2*sin2)**2)
            l4 = np.sqrt(LL**2+l2**2 -2*LL*l2*cos2)
            l3 = np.sqrt(LL**2+l1**2-2*LL*l1*cos1)

            l4_dot_L = (-LL*l2*cos2 + LL**2)
            l2_dot_L = (LL*l2*cos2)

            gXY = gfEBbatch(l2v, Lv, l2, l4, gradientEE, gradientBB)*filter_batch(l4)
            gYX = gfEBbatch(l4v, Lv, l4, l2, gradientEE, gradientBB)*filter_batch(l4)

            l1_dot_l2 = (l1*cos1*l2*cos2) + (l1*sin1*l2*sin2)
            l2_dot_l3 = l2_dot_L-l1_dot_l2
            l5_dot_l1 = l1**2-l1_dot_l2
            l5_dot_l3 = LL*l1*cos1-l2_dot_L-l1**2+l1_dot_l2

            cl5_XY = uEE(l5)*filter_batch(l5) #uTT(l5)*filter_batch(l5)
            Cl2 = gradientEE(l2)*filter_batch(l2) #uTT(l2)*filter_batch(l2)


            hX_l5_l2 = lmbdacos(l5, l2, l5v[0, :], l5v[1, :], l2v[0, :], l2v[1, :])
            hY_l5_l4 = lmbdasin(l5, l4, l5v[0, :], l5v[1, :], l4v[0, :], l4v[1, :])

            hY_l2_l4 = lmbdasin(l2, l4, l2v[0, :], l2v[1, :], l4v[0, :], l4v[1, :])
            hX_l2_l4 = lmbdacos(l2, l4, l2v[0, :], l2v[1, :], l4v[0, :], l4v[1, :])

            productA1 = -l5_dot_l1*l5_dot_l3*hX_l5_l2*hY_l5_l4*cl5_XY*gXY
            productC1 = l2_dot_l3*l1_dot_l2*(gXY*hY_l2_l4+gXY*hX_l2_l4)*Cl2*1/2

            #born_term = bispectrum_Born(l1, l3, LL)*8/(l1*l3*LL)**2
            born_term = 0.

            #ff = 1+(1-Wsinterpiter1(l1))*(1-Wsinterpiter1(l3))
            bispectrum_result = b3n.bispec_phi_general(l1, l3, LL, index)#*ff #*(1+(1-rho2interp1(l1))*(1-rho2interp1(l3)))

            bispectrum_total = bispectrum_result+born_term

            common = l1*l2*bispectrum_result/(2*np.pi)**4
            common_total = l1*l2*bispectrum_total/(2*np.pi)**4

            #A1 = common*productA1
            #C1 = common*productC1
            result = (productA1+productC1)*common
            result_total = (productA1+productC1)*common_total
            #return [common*productA1, common*productC1]
            return {"B": result, "PB": result_total} #{'A1': common*productA1, 'C1': common*productC1}
        
        result = integ(integrand, nitn = 2*nitn, neval = 2*neval)
        for k in keys:
            NTOT[k] += [result[k].mean]
        
        #NA1 += [result['A1'].mean]
        #NC1 += [result['C1'].mean]
    for k in keys:
        NTOT[k] = np.array(NTOT[k])
    results_n32[model] = NTOT