import pathlib
import shutil

def copy_mask_files(from_dir, human_mask_dir, machine_mask_dir, keyword="**/*oldwf.png"):
    #for holly's stuff mask_oldwf is Willie's mask
    #mask.png is the human annotated mask

    from_dir = pathlib.Path(from_dir)
    human_mask_dir = pathlib.Path(human_mask_dir)
    machine_mask_dir = pathlib.Path(machine_mask_dir)

    for mmask in from_dir.glob(keyword):
        #get corresponding machine mask
        folder, worm_id = mmask.parts[-2:]
        worm_id = worm_id.split(' ')[0]
        #print(folder+"/"+worm_id+"*mask.png")
        hmask = list(from_dir.glob(folder+"/"+worm_id+"*bf.png"))[0]
        
        #if folder isn't made then add it
        hmask_out_dir = human_mask_dir.joinpath(folder)
        mmask_out_dir = machine_mask_dir.joinpath(folder)
 
        if not hmask_out_dir.exists():
            hmask_out_dir.mkdir()
        if not mmask_out_dir.exists():
            mmask_out_dir.mkdir()

        #copy files over
        print("copying "+str(hmask) +" to "+str(hmask_out_dir))
        #shutil.copy(hmask, hmask_out_dir)
        print("copying "+str(mmask) +" to "+str(mmask_out_dir))
        shutil.copy(hmask, mmask_out_dir)
        
def copy_non_old_masks(from_dir, human_mask_dir, machine_mask_dir, keyword)
    """Need to copy 
    """
    from_dir = pathlib.Path(from_dir)
    human_mask_dir = pathlib.Path(human_mask_dir)
    machine_mask_dir = pathlib.Path(machine_mask_dir)

    for mmask in from_dir.glob(keyword):
        #get corresponding machine mask
        folder, worm_id = mmask.parts[-2:]
        worm_id = worm_id.split(' ')[0]
        #print(folder+"/"+worm_id+"*mask.png")
        hmask = list(from_dir.glob(folder+"/"+worm_id+"*bf.png"))[0]
        
        #if folder isn't made then add it
        hmask_out_dir = human_mask_dir.joinpath(folder)
        mmask_out_dir = machine_mask_dir.joinpath(folder)
 
        if not hmask_out_dir.exists():
            hmask_out_dir.mkdir()
        if not mmask_out_dir.exists():
            mmask_out_dir.mkdir()

        #copy files over
        print("copying "+str(hmask) +" to "+str(hmask_out_dir))
        #shutil.copy(hmask, hmask_out_dir)
        print("copying "+str(mmask) +" to "+str(mmask_out_dir))
        shutil.copy(hmask, mmask_out_dir)