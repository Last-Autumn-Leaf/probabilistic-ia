import matplotlib.pyplot as plt
import numpy
import numpy as np

def exo1(N,showplot=False):
    assert (type(N) == int )
    mean = [3, -1]
    cov = [[1, 0], [0, 1]]  # diagonal covariance
    m_=[]
    c_=[]
    #x, y = np.random.default_rng().multivariate_normal(mean, cov, 20).T
    for i in range(N):
        # here the question is not clear
        pts = np.random.default_rng().multivariate_normal(mean, cov, size=N)
        m_.append(pts.mean(axis=0))
        c_.append(np.cov(pts,ddof=1,rowvar=False))
        if showplot:
            plt.plot(pts[:, 0], pts[:, 1], '.', alpha=0.5)

    m_mean= np.mean(m_,axis=0)
    m_std= np.std(m_,axis=0,ddof=1)
    c_mean , c_std= np.mean(c_,axis=0) ,np.std(c_,axis=0,ddof=1)
    if showplot :
        plt.axis('equal')
        plt.grid()
        plt.title(f"N={N}")
        plt.show()
        #print("moyenne =",pts.mean(axis=0))
        #print("covariance =",np.cov(pts.T))
        print("Moyenne et écart-type de m:\n",m_mean,'\n',m_std)
        print("Moyenne et écart-type de cov:\n",c_mean,'\n',c_std)
    return ( (m_mean,m_std) , (c_mean,c_std) )

if __name__=='__main__':
    np.random.seed(0)
    N=[10,20,50,100,500]

    std_M=[]
    std_Cov=[]
    for n in N:
        m,cov=exo1(n)
        std_M.append(m[1]) # store the std for the mean
        std_Cov.append(cov[1]) # store the std for the Cov

    plt.plot(np.array([N,N]).T,std_M)
    plt.legend(['σm_1', 'σm_2'])
    plt.title('Variation des σm_X en fonction de N')
    plt.xlabel("Nombre d'itérations N")
    plt.ylabel("écart-type de la moyenne σm_X")
    plt.show()

    std_Cov=np.array(std_Cov).reshape((5,4))# 5X4

    plt.plot(np.array([N]*4).T, std_Cov, )
    plt.legend(['σΣ_'+str(i) for i in range(1,5)])
    plt.title('Variation des σΣ_X en fonction de N')
    plt.xlabel("Nombre d'itérations N")
    plt.ylabel("écart-type de la covariance σΣ_X")
    plt.show()