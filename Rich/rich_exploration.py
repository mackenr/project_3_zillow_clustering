from rich_acquisition import *
import seaborn as sns
from IPython.display import display
import matplotlib
import sklearn




def interestingdictionary(df,interestingcols,target,n=100):
    '''
    does some cool group by things to isolate a target variable by interestingcols catergorical cols, it then groups by all the uniquie values in that col and if the length is greater than n it saves it as dataframe and attaches it to a dictionary
    the return is dictionary of dictionaries where each unique col is the outer dictionary key and each unique val is the inner dictionary key






    '''
    outerdict={}
    for x,i in enumerate(interestingcols):
        
        uniques=df[i].unique().tolist()


        grp = df.groupby(i)   
    
    
        innerdict={}
        for y,j in enumerate(uniques):
            if  pd.isna(j)==False :
            
                group1=grp.get_group(j)
                isolated=(group1[target])
                if len(isolated)>n:
                    isolated=pd.DataFrame(isolated)   
                    innerdict.update({f'{df[i].name}_{y}':isolated})
                    # print(f'Working with {i}{j}')

                else:
                    # print('Not our condition')
                    continue
            else:

                    continue
        outerdict.update({i:innerdict})
        # print(f'outdict:\n\n{outerdict}')
    return outerdict

alpha=0.05
target='logerror'

def granulartwocombocomparison(df,interestingcols,target,n=100):
    '''
    This is to analyize for consistency. I don't think it is dynamic enought but it is a start. It would make sense to try different binning sizzes as we could find the optimal bins where the populations are mostly different.
    A simular implmentation might be useful just to compare each groupby to the overall population. 


    '''
    interestingAF=interestingdictionary(df,interestingcols,target,n)
    interestingAFkeys=list(interestingAF.keys())
    interestingAFtwoComboDict={}
    for i in range(0,len(interestingAFkeys)):
        interestingAFtwoComboDict.update({interestingAFkeys[i]:list(combinations(list(interestingAF.get(interestingAFkeys[i]).keys()),2))})

    interestingAFtwoComboDict.keys()
    samepopdict={}
    diffpopdict={}





    
    
    
    for i in interestingAFkeys:
        samepop=[]
        diffpop=[]
        combolist=interestingAFtwoComboDict.get(i)
        for j in combolist:
            a=interestingAF.get(i).get(j[0])
           
            b=interestingAF.get(i).get(j[1])
    
        
        
            data = pd.concat([a.assign(frame=f'{j[0]}'),
                          b.assign(frame=f'{j[1]}')])
           
            
            
            # g.add_legend(f'We compare:\n{str(i[0])} vs {str(i[1])}')
            varA=a[target]
            varB=b[target]
            t,p =stats.levene(varA,varB)
            if p <= alpha:
                varequal=False
            else:
                varequal=True
    
    
    
    
                       
            t,p = stats.ttest_ind(a[target], b[target],equal_var=varequal)
            nullsym=symbols('H_{0}')
            rejnull=symbols('Reject~H_{0}~?')
            null='The null hypothesis is that our populations are statistically the same.'

            
            if p / 2 > alpha:

                # equalpopstring=f'No, we observe that {j[0]} and {j[1]} are statistically the same:'
                # display(nullsym,null,rejnull)
                # print(equalpopstring)
                # print("We fail to reject our null")
                samepop.append(f'{j[0]},{j[1]}')
            elif t < 0:
            
                # equalpopstring=f'No, we observe that {j[0]} and {j[1]} are statistically the same:'
                # display(nullsym,null,rejnull)
                # print(equalpopstring)
                # print("We fail to reject our null")
                samepop.append(f'{j[0]},{j[1]}')
            else:        
                # equalpopstring=f'Yes, we observe that {j[0]} and {j[1]} are statistically different'
                # display(nullsym,null,rejnull)
                # print(equalpopstring)
                # print("We reject our null")
                diffpop.append(f'{j[0]},{j[1]}')
        samepopdict.update({i:samepop})
        diffpopdict.update({i:diffpop})
        
        samepopdict.update({i:samepop})
        diffpopdict.update({i:diffpop})
        

               
                
        
    
            # display(rejnull,equalpopstring)
    lengtha=sum([len(x) for x in [i for i in samepopdict.values()]])
    lengthb=sum([len(x) for x in [i for i in diffpopdict.values()]])
    print(f'different 2-combos count:{lengthb}\nvs\nsame 2-combos count:{lengtha} ')
    
    
    return samepopdict,diffpopdict,lengtha,lengthb


def scale_and_concat(df,scaler,scaled_vars):
    # the variables that still need scaling
   
    # create new column names for the scaled variables by adding 'scaled_' to the beginning of each variable name 
    scaled_column_names = ['scaled_' + i for i in scaled_vars]
    
    scaled_array = scaler.transform(df[scaled_vars])
    scaled_df = pd.DataFrame(scaled_array, columns=scaled_column_names, index=df.index.values)
    return pd.concat((df, scaled_df), axis=1)





# # select the X partitions: [X_train, X_validate, X_test]
# partitions= partitionslist[0:4]




def partitionslist_with_scaled(scaled_vars = ['latitude', 'longitude', 'bathroomcnt', 'taxrate']):
    
    partitionslist=partitionedZillowbyCounty()
    newpartitionslist=[]

    for i in partitionslist:
        countylist=[]
        countylist.append(i[0])
        X=i[1:4]
        X_train = X[0]
        scaler = MinMaxScaler(copy=True).fit(X_train[scaled_vars])
        for j in X:
            j = scale_and_concat(j,scaler,scaled_vars)
            countylist.append(j)
        countylist.append(i[-3])
        countylist.append(i[-2])
        countylist.append(i[-1])
        newpartitionslist.append(countylist)
    return newpartitionslist
# fit the minmaxscaler to X_train




def find_k(X_train, cluster_vars, k_range):
    sse = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k)

        # X[0] is our X_train dataframe..the first dataframe in the list of dataframes stored in X. 
        kmeans.fit(X_train[cluster_vars])

        # inertia: Sum of squared distances of samples to their closest cluster center.
        sse.append(kmeans.inertia_) 

    # compute the difference from one k to the next
    delta = [round(sse[i] - sse[i+1],0) for i in range(len(sse)-1)]

    # compute the percent difference from one k to the next
    pct_delta = [round(((sse[i] - sse[i+1])/sse[i])*100, 1) for i in range(len(sse)-1)]

    # create a dataframe with all of our metrics to compare them across values of k: SSE, delta, pct_delta
    k_comparisons_df = pd.DataFrame(dict(k=k_range[0:-1], 
                             sse=sse[0:-1], 
                             delta=delta, 
                             pct_delta=pct_delta))

    # plot k with inertia
    plt.plot(k_comparisons_df.k, k_comparisons_df.sse, 'bx-')
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.title('The Elbow Method to find the optimal k\nFor which k values do we see large decreases in SSE?')
    plt.show()

    # plot k with pct_delta
    plt.plot(k_comparisons_df.k, k_comparisons_df.pct_delta, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Percent Change')
    plt.title('For which k values are we seeing increased changes (%) in SSE?')
    plt.show()

    # plot k with delta
    plt.plot(k_comparisons_df.k, k_comparisons_df.delta, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Absolute Change in SSE')
    plt.title('For which k values are we seeing increased changes (absolute) in SSE?')
    plt.show()

    return k_comparisons_df



def create_clusters(X_train, k, cluster_vars):
    # create kmean object
    kmeans = KMeans(n_clusters=k, random_state = 13)

    # fit to train and assign cluster ids to observations
    kmeans.fit(X_train[cluster_vars])

    return kmeans





# get the centroids for each distinct cluster...

def get_centroids(kmeans, cluster_vars, cluster_name):
    # get the centroids for each distinct cluster...

    centroid_col_names = ['centroid_' + i for i in cluster_vars]

    centroid_df = pd.DataFrame(kmeans.cluster_centers_, 
                               columns=centroid_col_names).reset_index().rename(columns={'index': cluster_name})

    return centroid_df



# label cluster for each observation in X_train (X[0] in our X list of dataframes), 
# X_validate (X[1]), & X_test (X[2])

def assign_clusters(kmeans, cluster_vars, cluster_name, centroid_df, X):
    for i in range(len(X)):
        clusters = pd.DataFrame(kmeans.predict(X[i][cluster_vars]), 
                            columns=[cluster_name], index=X[i].index)

        clusters_centroids = clusters.merge(centroid_df, on=cluster_name, copy=False).set_index(clusters.index.values)

        X[i] = pd.concat([X[i], clusters_centroids], axis=1)
    return X


def pearsonsRsquareLogError(df,a):
    b='logerror'
    i=[a,b]
    
    r,p=stats.pearsonr(df[i[0]],df[i[1]])
    nullsym=symbols('H_{0}')
    rejnull=symbols('Reject~H_{0}~?')
    null='There is zero correlation according to Pearson R '
    if p > alpha:        
        corstring=f'No, we observe that {i[0]} and {i[1]} show no correlation by pearsons R:'
        display(nullsym,null,rejnull)
        print(f"{corstring}\nHence, we fail to reject our null hypothesis\n\n")
    
    else:      
        corstring=f'Yes, we observe that {i[0]} and {i[1]} show observable correlation by pearsons R:'
        display(nullsym,null,rejnull)
        print(f"{corstring}\nOur r value is:{r}\nHence, we reject our null hypothesis\n\n")



def threeQandA_stats_viz_counties(train_la,orange_train,ventura_train):
    traincountyset=set(['train_la','orange_train','ventura_train'])
    traincounty2combos=list(combinations(traincountyset,2))
    traincountyset=list(traincountyset)
    
    traincountydf=[train_la,orange_train,ventura_train]
    
    mapper=dict(zip(traincountyset,traincountydf))
    
    
    for i in traincounty2combos:
    
    
        a=mapper.get(i[0])
        b=mapper.get(i[1])
        data = pd.concat([a.assign(frame=f'{i[0]}'),
                      b.assign(frame=f'{i[1]}')])
        g=sns.FacetGrid(data, col='frame',hue='frame')
    
        g.map(sns.histplot,'logerror',kde=True)
        plt.show()
        # g.add_legend(f'We compare:\n{str(i[0])} vs {str(i[1])}')
        varA=a[target]
        varB=b[target]
        t,p =stats.levene(varA,varB)
        if p < alpha:
            varequal=False
        else:
            varequal=True
        
    
    
        
        t,p = stats.ttest_ind(a[target], b[target],equal_var=varequal)
        nullsym=symbols('H_{0}')
        rejnull=symbols('Reject~H_{0}~?')
        null='The null hypothesis is that our populations are statistically the same.'
    
        if p / 2 > alpha:
            
            equalpopstring=f'No, we observe that {i[0]} and {i[1]} are statistically the same:'
            display(nullsym,null,rejnull)
            print(f"{equalpopstring}\nHence, we fail to reject our null hypothesis\n\n")
        elif t < 0:
        
            equalpopstring=f'No, we observe that {i[0]} and {i[1]} are statistically the same:'
            display(nullsym,null,rejnull)
            print(f"{equalpopstring}\nHence, we fail to reject our null hypothesis\n\n")
        else:
            
            equalpopstring=f'Yes, we observe that {i[0]} and {i[1]} are statistically different'
            display(nullsym,null,rejnull)
            print(f"{equalpopstring}\nHence, we reject our null hypothesis\n\n")  




def county_train_pie(train_la,orange_train,ventura_train):
    data = [len(train_la),len(orange_train),len(ventura_train)]
    labels = ['LA','OC','Ventura']

    #define Seaborn color palette to use
    colors = sns.color_palette('pastel')

    #create pie chart
    plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%')
    plt.show()      



def ageandlogrelpie(lengthsame,lengthdiff):
    #define data
    data = [lengthsame, lengthdiff]
    labels = ['Same pop', 'Different pop']
    
    #define Seaborn color palette to use
    colors = sns.color_palette('pastel')
    
    #create pie chart
    plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%')
    plt.show()