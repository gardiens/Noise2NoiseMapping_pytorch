import torch 
from tqdm import tqdm 
import time 
try: 
    from .results.display import display_result
    from .loss import * 
except:
    pass 

def evaluate_loss_cst_vf(
    net,
    pc,
    # normals,
    hints_pc=None,
    gtsdf=None,
    # vf,
    list_loss=[],
    lambda_pc=1,
    lambda_eik=2,
    lambda_hint=1,
    lambda_lse=1,
    batch_size=2000,
    lambda_gc=1,
    lambda_emd=1,
    dim_space=2,
    lambda_gc2=0,
):
    loss=0
    pc.requires_grad = True
    # they are all at time 0 ! 
    # Compute the EMD 
    # loss_EMD=compute_EMD(net=net,input_pc=pc,batch_size=batch_size)
    # Il faut voir ça comme la fonction qui "définit la surface"
    if lambda_emd>0:
        loss_EMD=compute_EMD_geomloss(net=net,input_pc=pc,batch_size=batch_size,loss=EMD_loss())
        list_loss["emd"].append(float(loss_EMD))
        loss+=    lambda_emd * loss_EMD


    if lambda_pc>0:
        loss_pc=loss_shape_data(net=net,pc=pc,batch_size=batch_size)
        list_loss["pc"].append(float(loss_pc))
        loss+=lambda_pc * loss_pc
    # Compute the geometric consistency loss
    # c'est la partie du coup qui va régulariser en dhors du nuage dep oints !
    if lambda_gc>0:
        # loss_gc=loss_geometric_consistency(net=net,input_pc=pc)

        loss_gc=loss_geometric_consistency2(net=net,input_pc=pc,dim_space=pc.shape[-1])
        loss+=lambda_gc * loss_gc
        list_loss["gc"].append(float(loss_gc))

    if lambda_gc2:
        loss_gc2=loss_geometric_consistency3(net=net,input_pc=pc,dim_space=pc.shape[-1])
        loss+=lambda_gc2 * loss_gc2
    # compute and store standard losses
    # loss_pc = loss_shape_data(net, pc, normals, batch_size)
    if not hints_pc is None and lambda_hint>0:
        loss_hint = loss_amb(net, hints_pc, gtsdf, batch_size)
        list_loss["hint"].append(float(loss_hint))
        loss+=lambda_hint * loss_hint
    if lambda_eik>0:
        loss_eik = loss_eikonal(net, batch_size,dim_space=dim_space)
        list_loss["eik"].append(float(loss_eik))
        loss+=lambda_eik * loss_eik
    # loss_lse = loss_lse_eq(net, vf, batch_size)

    # append all the losses
    # lpc.append(float(loss_pc))
    
    # leik.append(float(loss_eik))
    # lemd.append(float(loss_EMD))
    # lgc.append(float(loss_gc))
    # lh.append(float(loss_hint))
    # llse.append(float(loss_lse))

    # sum the losses of reach of this set of points
    # loss = (
    #     # lambda_pc * loss_pc
    #     # lambda_eik * loss_eik+

    #     # + lambda_hint * loss_hint
    #     # + lambda_lse * loss_lse
    # )
    return loss
def optimize_nise_vf(
    net,
    pc0,
    # nc0,
    hints0=None,
    gtsdf0=None,
    # vf,
    # lpc,
    # leik,
    # lh,
    # llse,
    # lemd,
    # lgc,
    lambda_pc=1,
    lambda_eik=2,
    lambda_hint=1,
    lambda_lse=2,
    lambda_gc=1,
    lambda_emd=1,
    lambda_gc2=0,
    batch_size=2000,
    nepochs=100,
    plot_loss=True,
    list_loss=[],
    follow_paper=False
):
    # pc0, nc0, hints0, gtsdf0 are the input data at time  0
    # vf is the velocity field
    # lpc, leik, lh, llse are lists to store the losses
    # lambda_pc, lambda_eik, lambda_hint, lambda_lse are the weights of the losses
    optim = torch.optim.Adam(params=net.parameters(), lr=1e-4)
    print(f"starting loss, lambda_pc={lambda_pc}, lambda_eik={lambda_eik}, lambda_hint={lambda_hint}, lambda_lse={lambda_lse}, lambda_gc={lambda_gc}, lambda_emd={lambda_emd}")
    if follow_paper:
        print("beware HARDCODED version")
    tinit = time.time()
    pbar = tqdm(total=nepochs, desc="Training")
    marqueur=True 
    marqueur_gc=True
    marqueur_pc=True
    marqueur_emd=True
    marqueur_gc2=True
    for batch in range(nepochs):
        optim.zero_grad()
        if lambda_gc ==0:
            if marqueur_gc:
                print("lambda_gc is null")
                marqueur_gc=False 
        if lambda_emd ==0:
            if marqueur_emd:
                print("lambda_emd is null")
                marqueur_emd=False
        if lambda_pc ==0:
            if marqueur_pc:
                print("lambda_pc is null")
                marqueur_pc=False
        if lambda_eik ==0:
            if marqueur:
                print("lambda_eik is null")
                marqueur=False
        if follow_paper:
            # print("beware HARDCODED version")
            
            if batch<1000:
                lambda_gc=0.1*1/1500*(1500-batch)
            else:
                lambda_gc=0

        loss = evaluate_loss_cst_vf(
            net=net,
            pc=pc0,
            # normals=nc0,
            hints_pc=hints0,
            gtsdf=gtsdf0,
            # vf=vf,
            # lpc=lpc,
            # leik=leik,
            # lemd=lemd,
            # lgc=lgc,
            list_loss=list_loss,
            lambda_pc=lambda_pc,
            lambda_eik=lambda_eik,
            lambda_hint=lambda_hint,
            lambda_lse=lambda_lse,
            lambda_gc2=lambda_gc2,
            batch_size=batch_size,
            lambda_gc=lambda_gc,
            lambda_emd=lambda_emd,
            dim_space=pc0.shape[1],

        )
        loss.backward()
        optim.step()
        if batch % 100 == 99 or batch == 0:
            # print(f"Epoch {batch}/{nepochs} - loss : {loss.item()}")
            pbar.set_postfix({"loss": loss.item()})
        pbar.update(1)
        if batch %2000==0 and plot_loss:
            display_result(net, resolution=200, figsize=(14, 5))
    tend = time.time()

    print("Optimizing NN took", "{:.2f}".format(tend - tinit), "s.")