import numpy as np


class TxRx:
    def __init__(self, txrx_type, is_transmitter, is_receiver, pos, name, id=None, grid_side=None, grid_spacing=None):
        self.txrx_name = name
        self.txrx_id = id
        self.txrx_type = txrx_type
        self.is_transmitter = is_transmitter
        self.is_receiver = is_receiver
        self.txrx_pos = pos
        self.grid_side = grid_side
        self.grid_spacing = grid_spacing

# FIXME the editor only support tx_power = 0
class TxRxEditor:
    def __init__(self, infile_path=None, template_path="resource/template/txrx"):
        self.infile_path = infile_path

        self.txrx = []
        self.template_path = template_path

        self.txrx_file = None
        self.parse()
        self.txpower = 0
        

    def read_tx_power(self):
        with open(self.infile_path, "r") as f:
            txrx_file = f.readlines()
        power_vals = []
        for line in txrx_file:
            if line.split(" ")[0] == "power":
                power_vals.append(np.float64(line.split(" ")[1]))
        if len(set(power_vals)) > 1:
            print(
                "Power of the transmitters are not uniform. The generator does not support such design!"
            )
        if len(power_vals) == 0:
            print("Cannot find transmission power value in .txrx file!")
        return power_vals[0]


    def parse(self):
        try:
            with open(self.infile_path, "r") as f:
                txrx_file = f.read().splitlines()
        except:
            return
        for i, line in enumerate(txrx_file):
            if ("begin_<points>" in line) or ("begin_<grid>" in line):
                txrx_name = txrx_file[i].split(" ")[1]
                txrx_id = int(txrx_file[i + 1].split(" ")[1])
                txrx_type = line.split("<")[1].split(">")[0]
                grid_side = None
                grid_spacing = None
            if "nVertices" in line:
                pos = np.asarray(
                    txrx_file[i + 1].split(" "), np.float64
                )
            if "is_transmitter" in line:
                is_transmitter = True if (line.split(" ")[1] == "yes") else False
            if "is_receiver" in line:
                is_receiver = True if (line.split(" ")[1] == "yes") else False
            if "side1" in line:
                side1 = np.float64(txrx_file[i].split(" ")[1])
                side2 = np.float64(txrx_file[i + 1].split(" ")[1])
                grid_spacing = np.float64(txrx_file[i + 2].split(" ")[1])
                grid_side = np.asarray([side1, side2])
            if ("end_<points>" in line) or ("end_<grid>" in line):
                self.txrx.append(TxRx(txrx_type, is_transmitter, is_receiver, pos, txrx_name, txrx_id, grid_side, grid_spacing))


    #TODO this functin has not been tested
    def add_txrx(self, txrx_type, is_transmitter, is_receiver, pos, name, id=None, grid_side=None, grid_spacing=None):
        if not id:
            try:
                id = self.txrx[-1].txrx_id + 1
            except:
                id = 1
        new_txrx = TxRx(txrx_type, is_transmitter, is_receiver, pos, name, id=id, grid_side=grid_side, grid_spacing=grid_spacing)
        self.txrx.append(new_txrx)


    #TODO this functin has not been tested
    def remove_txrx(self, id=None):
        if not id:
            self.txrx = []
            return
        
        pop_idx = []
        for i, x in enumerate(self.txrx):
            if x.txrx_id in id:
                pop_idx.append(i)
        pop_idx = pop_idx[::-1]
        for p in pop_idx:
            self.txrx.pop(p)


    def save(self, outfile_path):
        # clean the output file before writing
        open(outfile_path, "w+").close()
        for x in self.txrx:
            with open(self.template_path+"/"+x.txrx_type+".txt", "r") as f:
                template = f.readlines()
            for i, line in enumerate(template):
                if ("begin_<points>" in line) or ("begin_<grid>" in line):
                    template[i] = "begin_<%s> %s\n" % (x.txrx_type, x.txrx_name)
                    template[i+1] = "project_id %d\n" % x.txrx_id

                if "nVertices" in line:
                    template[i+1] = "%.15f %.15f %.15f\n" % (x.txrx_pos[0], x.txrx_pos[1], x.txrx_pos[2])
                
                if "is_transmitter" in line:
                    tmp = "yes" if x.is_transmitter else "no"
                    template[i] = "is_transmitter %s\n" % tmp

                if "is_receiver" in line:
                    tmp = "yes" if x.is_receiver else "no"
                    template[i] = "is_receiver %s\n" % tmp

                if "side1" in line:
                    template[i] = "side1 %.5f\n" % np.float32(x.grid_side[0])
                    template[i+1] = "side2 %.5f\n" % np.float32(x.grid_side[1])
                    template[i+2] = "spacing %.5f\n" % np.float32(x.grid_spacing)
                if line.split(" ")[0] == "power":
                    template[i] = "power %.5f\n" % self.txpower
            with open(outfile_path, "a") as out:
                out.writelines(template)


if __name__ == "__main__":
    outfile_path = "test/gwc_test.txrx"
    editor = TxRxEditor()
    # editor.remove_txrx()
    editor.add_txrx("points", True, True, [0,0,6], "BS")
    editor.add_txrx("grid", False, True, [-187,-149,2], "UE_grid", grid_side=[379,299], grid_spacing=5)
    editor.save(outfile_path)
    print("done")