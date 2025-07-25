#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 12:02:41 2018

@author: isaaclera
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

time = 1000000
pathSimple = "exp_rev/"

# =============================================================================
# ALERTA: PREPARA LOS RESULTADOS - LOS GRAFICOS SON GENERADOS en 2b
# =============================================================================


# =============================================================================
# Computes the Response time by (App,User)
# It takes into account the number of messages (mode) that it sends each (App,User)
# =============================================================================
def getRbyApp(df, dtmp):
    myDeadlines = [
        487203.22,
        487203.22,
        487203.22,
        474.51,
        302.05,
        831.04,
        793.26,
        1582.21,
        2214.64,
        374046.40,
        420476.14,
        2464.69,
        97999.14,
        2159.73,
        915.16,
        1659.97,
        1059.97,
        322898.56,
        1817.51,
        406034.73,
    ]

    dr = pd.DataFrame(
        columns=["app", "user", "avg", "std", "m", "r", "invalid", "over"]
    )  # m - numero de mensajes enviados
    times = []
    ixloc = 0
    for g in dtmp.keys():
        ids = dtmp[g]
        responses = []
        messages = []
        over = 0
        # Firstly, it computes the mode in all the app,user transmissions
        for i in ids:
            messages.append(
                df[df.id == i].shape[0]
            )  # number of messages send by the user

        # Requests with a inferior number of messages are filtered
        msg = np.array(messages)
        mode = stats.mode(msg).mode[0]

        # Secondly, if each transmission has the same mode then the time is storaged
        invalid = 0
        for i in ids:
            dm = df[df.id == i]
            if mode == dm.shape[0]:
                r = dm["time_out"].max() - dm["time_emit"].min()
                if r <= myDeadlines[g[0]]:
                    #                if True:
                    responses.append(r)
                    times.append(dm["time_emit"].min())
                else:
                    over += 1
            else:
                invalid += 1

        resp = np.array(responses)

        avg = resp.mean()
        dsv = resp.std()
        dr.loc[ixloc] = [g[0], g[1], avg, dsv, mode, resp, invalid, over]
        ixloc += 1
        print(g, "\t", len(dtmp[g]), "\t", invalid, "\t", over)

    return dr, times


def drawBoxPlot_User_App(dr, app):
    fig, ax = plt.subplots()
    ax.boxplot(dr[dr.app == app]["r"].values)
    # TODO ILP CHANGE POSITION
    ax.set_xticklabels(dr[dr.app == app]["user"].values)
    ax.set_title("App: %i" % app)
    ax.set_ylabel("Time Response")
    ax.set_xlabel("User")
    plt.show()


def set_box_color(bp, color):
    plt.setp(bp["boxes"], color=color)
    plt.setp(bp["whiskers"], color=color)
    plt.setp(bp["caps"], color=color)
    plt.setp(bp["medians"], color=color)


### It computes the Response of each app
def getAllR(dr):
    dar = pd.DataFrame(columns=["app", "r"])
    ixloc = 0
    for k, g in dr.groupby(["app"]):
        values = np.array([])
        for item in g.values:
            values = np.concatenate((values, item[5]), axis=0)
        dar.loc[ixloc] = [k, values]
        ixloc += 1
    return dar


def drawBoxPlot_Both_USER(app, dr, drILP):
    fig, ax = plt.subplots()
    data_a = dr[dr.app == app].r.values
    data_b = drILP[drILP.app == app].r.values
    ticks = list(np.sort(dr[dr.app == app].user.unique()))
    bpl = plt.boxplot(
        data_a, positions=np.array(range(len(data_a))) * 2.0 - 0.4, sym="", widths=0.6
    )
    bpI = plt.boxplot(
        data_b, positions=np.array(range(len(data_b))) * 2.0 + 0.4, sym="", widths=0.6
    )
    set_box_color(bpl, "#5ab4ac")  # colors are from http://colorbrewer2.org/
    set_box_color(bpI, "#d8b365")
    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c="#5ab4ac", label="Partition")
    plt.plot([], c="#d8b365", label="ILP")
    plt.legend()

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks) * 2)
    # plt.ylim(0, 10000)
    #    plt.ylim(00, 1000)
    ax.set_title("App: %i" % app)
    ax.set_ylabel("Time Response")
    ax.set_xlabel("User")
    plt.tight_layout()
    plt.savefig(pathSimple + "app%i.png" % app)


def drawBoxPlot_App(dar, darILP, labeldar="Partition", labelILP="ILP"):
    fig, ax = plt.subplots()
    # This is not work :/
    # data_a = dr.groupby(["app"]).agg({"values": lambda x: list(x.sum())})
    data_a = dar.r.values
    data_b = darILP.r.values
    ticks = list(np.sort(dar.app.unique()))

    bpl = plt.boxplot(
        data_a, positions=np.array(range(len(data_a))) * 2.0 - 0.4, sym="", widths=0.6
    )
    bpI = plt.boxplot(
        data_b, positions=np.array(range(len(data_b))) * 2.0 + 0.4, sym="", widths=0.6
    )
    set_box_color(bpl, "#5ab4ac")  # colors are from http://colorbrewer2.org/
    set_box_color(bpI, "#d8b365")
    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c="#5ab4ac", label=labeldar)
    plt.plot([], c="#d8b365", label=labelILP)
    plt.legend()

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks) * 2)
    # plt.ylim(50, 400)
    # plt.ylim(0, 10000)
    ax.set_title("All Apps")
    ax.set_ylabel("Time Response")
    ax.set_xlabel("App")
    plt.tight_layout()


path = pathSimple + "Results__%s" % time
pathILP = pathSimple + "Results_ILP_%s" % time
NEW = False

# =============================================================================
# Normal loads
# =============================================================================
df = pd.read_csv(path + ".csv")
dtmp = df[df["module.src"] == "None"].groupby(["app", "TOPO.src"])["id"].apply(list)
dr, timeC = getRbyApp(df, dtmp)
drAll = getAllR(dr)

dfILP = pd.read_csv(pathILP + ".csv")
dtmp2 = (
    dfILP[dfILP["module.src"] == "None"].groupby(["app", "TOPO.src"])["id"].apply(list)
)
drILP, timeILP = getRbyApp(dfILP, dtmp2)
drAllILP = getAllR(drILP)

# =============================================================================
# FAILs CENTRALITY load
# =============================================================================
# df2 = pd.read_csv(pathSimple+"Results_FAIL__%s.csv"%time)
# dtmp3 = df2[df2["module.src"]=="None"].groupby(['app','TOPO.src'])['id'].apply(list)
# drFAIL,timesFail = getRbyApp(df2,dtmp3)
# drAllFAIL = getAllR(drFAIL)
#
# df3 = pd.read_csv(pathSimple+"Results_FAIL_ILP_%s.csv"%time)
# dtmp4 = df3[df3["module.src"]=="None"].groupby(['app','TOPO.src'])['id'].apply(list)
# drILPFAIL,timesILPFail = getRbyApp(df3,dtmp4)
# drAllILPFAIL = getAllR(drILPFAIL)


# =============================================================================
# FAILs RANDOM load
# =============================================================================
df4 = pd.read_csv(pathSimple + "Results_RND_FAIL__%s.csv" % time)
dtmp5 = df4[df4["module.src"] == "None"].groupby(["app", "TOPO.src"])["id"].apply(list)
drFAILR, timesFailR = getRbyApp(df4, dtmp5)
drAllFAILR = getAllR(drFAILR)

df5 = pd.read_csv(pathSimple + "Results_RND_FAIL_ILP_%s.csv" % time)
dtmp6 = df5[df5["module.src"] == "None"].groupby(["app", "TOPO.src"])["id"].apply(list)
drILPFAILR, timesILPFailR = getRbyApp(df5, dtmp6)
drAllILPFAILR = getAllR(drILPFAILR)


# =============================================================================
# DEADLINES
# =============================================================================

deadline = {}
# for app in range(0,20):
#    print "-"*30
#    print "APP %i" %app
#    val1 = drAll[drAll["app"]==app]["r"].max().max()
#    val2 = drAllILP[drAllILP["app"]==app]["r"].max().max()
#    if val1 > val2:
#        print val1
#        print "MAX. Partition"
#        deadline[app]=val1
#    else:
#        print val2
#        print "MAX. ILP"
#        deadline[app]=val2
#
# for k in range(0,20):
#    valMax = deadline[k]
#    valC = drAllFAIL[drAllFAIL["app"]==k]["r"].max() ## Array
#    valILP = drAllILPFAIL[drAllILPFAIL["app"]==k]["r"].max() ## Array
#    x = valC[~np.isnan(valC)]
#    y = valILP[~np.isnan(valILP)]
#    print "APP %i"%k
#    print "\tPartition Total over deadline= %i / %i"%(np.sum(x>valMax), len(valC) )
#    print "\tILP    Total over deadline= %i / %i"%(np.sum(y>valMax), len(valILP) )
#


# =============================================================================
#  PLOTS: TIMES(request) under QoS along the simulation
# =============================================================================


## Centrality results
# dFailsC = pd.DataFrame(index=np.array(timesFail).astype('datetime64[s]'))
# dFailsC["QTY"]=np.ones(len(timesFail))
# dFailsC = dFailsC.resample('500s').agg(dict(QTY='sum'))
# QTYFails = dFailsC.QTY.values
#
#
# dFailsILP = pd.DataFrame(index=np.array(timesILPFail).astype('datetime64[s]'))
# dFailsILP["QTY"]=np.ones(len(timesILPFail))
# dFailsILP = dFailsILP.resample('500s').agg(dict(QTY='sum'))
# QTYFailsILP = dFailsILP.QTY.values

## Random results

dFailsCR = pd.DataFrame(index=np.array(timesFailR).astype("datetime64[s]"))
dFailsCR["QTY"] = np.ones(len(timesFailR))
dFailsCR = dFailsCR.resample("500s").agg(dict(QTY="sum"))
QTYFailsCR = dFailsCR.QTY.values


dFailsILPR = pd.DataFrame(index=np.array(timesILPFailR).astype("datetime64[s]"))
dFailsILPR["QTY"] = np.ones(len(timesILPFailR))
dFailsILPR = dFailsILPR.resample("500s").agg(dict(QTY="sum"))
QTYFailsILPR = dFailsILPR.QTY.values


# dC = pd.DataFrame(index=np.array(timeC).astype("datetime64[s]"))
# dC["QTY"] = np.ones(len(timeC))
# dC = dC.resample("500s").agg(dict(QTY="sum"))
# QTYC = dC.QTY.values

# dILP = pd.DataFrame(index=np.array(timeILP).astype("datetime64[s]"))
# dILP["QTY"] = np.ones(len(timeILP))
# dILP = dILP.resample("500s").agg(dict(QTY="sum"))
# QTYILP = dILP.QTY.values


#####ALERTA
# np.save("exp_final3/QTYC.npy", QTYC)
# np.save("exp_final3/QTYFailsCR.npy", QTYFailsCR)
# np.save("exp_final3/QTYFailsILPR.npy", QTYFailsILPR)
# dr.to_pickle("exp_final3/dr.pkl")
# drILP.to_pickle("exp_final3/drILP.pkl")


######


#### GRAFICAS MOVIDAS A idem-2b.py

# ticks = range(len(QTYC))
# ticksV = np.array(ticks)*10
#
### Unifiend length with 0 at the end
##QTYFails = np.concatenate((QTYFails,np.zeros(len(QTYC)-len(QTYFails))))
##QTYFailsILP = np.concatenate((QTYFailsILP,np.zeros(len(QTYC)-len(QTYFailsILP))))
# QTYFailsCR = np.concatenate((QTYFailsCR,np.zeros(len(QTYC)-len(QTYFailsCR))))
# QTYFailsILPR = np.concatenate((QTYFailsILPR,np.zeros(len(QTYC)-len(QTYFailsILPR))))
#
#
##fig, ax = plt.subplots(figsize=(16,8))
# fig, ax = plt.subplots(figsize=(32.0,8.0))
##ax1.plot(ticks, QTYFails, '-',color='#a8b3a5')
##ax1.plot(ticks, QTYFailsILP, '-',color='#d8b365')
# ax.plot(ticks, QTYC, '-',color='#756bb1',alpha=1.,linewidth=2)
##ax1.plot(ticks, QTYILP, '-') == Are near the same > partition
# ax.plot(ticks, QTYFailsCR, color='#a6bddb',alpha=1.,linewidth=2)
# ax.plot(ticks, QTYFailsILPR,color='#e34a33',alpha=1.,linewidth=2)
#
# z = np.polyfit(ticks, QTYC, 10)
# p = np.poly1d(z)
##ax1 = ax.plot(ticks,p(ticks),"-",color='#FFFFFF',linewidth=4)
# ax1 = ax.plot(ticks,p(ticks),":",color='#c1bcdc',linewidth=6,label="Total num. of requests",path_effects=[pe.Stroke(linewidth=8, foreground='purple'), pe.Normal()])
#
# idx = np.isfinite(QTYFailsCR) & np.isfinite(QTYFailsCR) #elementos con nan.>Error
# z1 = np.polyfit(np.array(ticks)[idx], np.array(QTYFailsCR)[idx], 10)
# p1 = np.poly1d(z1)
##ax2 = ax.plot(ticks,p1(ticks),"-",color='#FFFFFF',linewidth=4)
# ax2 = ax.plot(ticks,p1(ticks),"-",color='#a6bddb',linewidth=6,label="Partition",path_effects=[pe.Stroke(linewidth=8, foreground='#4a78b5'), pe.Normal()])
#
# idx = np.isfinite(QTYFailsILPR) & np.isfinite(QTYFailsILPR)
# z2 = np.polyfit(np.array(ticks)[idx], np.array(QTYFailsILPR)[idx], 10)
# p2 = np.poly1d(z2)
##ax3 = ax.plot(ticks,p2(ticks),"-",color='#FFFFFF',linewidth=4)
# ax3 = ax.plot(ticks,p2(ticks),"--",color='#f6c3bc',linewidth=6,label="ILP",path_effects=[pe.Stroke(linewidth=8, foreground='r'), pe.Normal()])
#
# ax.set_xlabel("Simulation time", fontsize=28)
# ax.set_ylabel("QoS satisfaction \n (num. of requests)", fontsize=28)
# ax.tick_params(labelsize=20)
# ax.set_xlim(-20,2020)
##plt.legend([ax1,ax2,ax3],['Total num. of requests','Partition','ILP'],loc="upper right",fontsize=18)
# plt.legend(loc="below left",fontsize=18)
# plt.tight_layout()
# plt.savefig('QSR-Random-32.pdf', format='pdf', dpi=600)
# plt.show()


#
## =============================================================================
## Boxplot matriz of each app - gtw/user
## =============================================================================
#
# def drawBoxPlot_Both_USER_ax(app,dr,drILP,ax):
#    data_a=dr[dr.app==app].r.values
#    data_b=drILP[drILP.app==app].r.values
#    ticks = list(np.sort(dr[dr.app==app].user.unique()))
#    bpl = ax.boxplot(data_a, positions=np.array(range(len(data_a)))*2.0-0.4, sym='', widths=0.55,
#                     whiskerprops = dict(linewidth=2),
#                    boxprops = dict(linewidth=2),
#                     capprops = dict(linewidth=2),
#                    medianprops = dict(linewidth=2))
#    bpI = ax.boxplot(data_b, positions=np.array(range(len(data_b)))*2.0+0.4, sym='', widths=0.55,
#                        whiskerprops = dict(linewidth=2),
#                    boxprops = dict(linewidth=2),
#                     capprops = dict(linewidth=2),
#                    medianprops = dict(linewidth=2))
#    set_box_color(bpl, '#a6bddb')
#    set_box_color(bpI, '#e34a33')
#    ax.get_xaxis().set_ticks(range(0, len(ticks) * 2, 2))
#    ax.set_xticklabels(ticks)
#    ax.set_xlim(-2, len(ticks)*2)
#    ax.plot([], c='#a6bddb', label="Partition",linewidth=3)
#    ax.plot([], c='#e34a33', label="ILP",linewidth=3)
#
#
# fig, axlist = plt.subplots(nrows=4, ncols=5, figsize=(14, 10))
# for idx,ax in enumerate(axlist.flatten()):
#    drawBoxPlot_Both_USER_ax(idx,dr,drILP,ax)
#
# fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.12)
# fig.subplots_adjust(hspace=0.4,wspace=0.35)
# axlist.flatten()[-2].legend(loc='upper center', bbox_to_anchor=(-0.85, -0.43), ncol=4,fontsize=16 )
#
# axlist[3][2].set_xlabel('IoT devices (Gateways id.)',fontsize=14)
# axlist[1][0].set_ylabel('Response time (ms)',fontsize=14)
# axlist[1][0].yaxis.set_label_coords(-0.4, 0)
# ax.tick_params(labelsize=12)
# plt.savefig('Boxplot.pdf', format='pdf', dpi=600)
# plt.show()
#
