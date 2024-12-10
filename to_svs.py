import openslide as ops
from datetime import datetime as dt
import wsic
import numpy as np


def run():

    relative_path_read = 'G:/placenta/'
    relative_path_write = 'G:/placenta_to_svs/'

    file_name = '12-S -027088-2'

    # wsi_file =  wsic.readers.OpenSlideReader(relative_path_read + file_name + '.mrxs')

    # print(type(wsi_file[0]))
    # mrxs_file = ops.open_slide(relative_path_read + file_name)
    # mrxs_file.read_region()
    #
    # img = self.os_slide.read_region(
    #     location=(start_x, start_y),
    #     level=0,
    #     size=(end_x - start_x, end_y - start_y),
    # )
    # return np.array(img.convert("RGB"))
    #
    # print(wsi_file)
    # writer = wsic.writers.SVSWriter(relative_path_write + file_name + '.svs', (1000,1000))
    # writer.copy_from_reader(wsi_file)
    # print(mrxs_file.properties)
    # print(mrxs_file.dimensions)
    #
    # size = 20000
    #
    # patho_img = mrxs_file.get_thumbnail([size, size])
    # patho_img.show()

def ops_run():
    relative_path_read = 'G:/placenta/'
    relative_path_write = 'G:/placenta_to_svs/'

    file_name = '12-S -027088-2'
    ops_file = ops.open_slide(relative_path_read + file_name + '.mrxs')
    print(ops_file.dimensions)
    size_x, size_y = ops_file.dimensions
    print(size_x, size_x//2)
    print(ops_file.get_thumbnail([size_x//8, size_y//8]))
    ops_file.get_thumbnail([size_x // 8, size_y // 8]).show()
    # arr = np.array(ops_file.get_thumbnail(ops_file.dimensions))
    # print(arr)



if __name__ == '__main__':
    t_begin = dt.now()
    print('Started at ', t_begin)

    ops_run()

    t_end = dt.now()
    print('Ended at ', t_end)
    print('Total Time :: ', t_end - t_begin)