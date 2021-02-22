function create_dataset(rel_path)

dir_in = strcat('../', rel_path);
dir_out = strcat('../','mod/', rel_path);
bg = 'bg.jpg';

bg = imread(bg) ;

l_size = 3 ;

dirs = dir(dir_in) ;
% Get a logical vector that tells which is a directory.
dirFlags = [dirs.isdir] ;
% Extract only those that are directories.
chr_dirs = dirs(dirFlags) ;

aleph = 'א' ;


for k = 1:length(chr_dirs)
    dir_name = chr_dirs(k).name ;
    if dir_name == '.' | dir_name == '..'
        continue
    end
    chr_dir_path = strcat(dir_in, dir_name) ;
    chr_output_dir_path = strcat(dir_out, dir_name) ;
    mkdir([dir_out dir_name]);
    files = dir(chr_dir_path) ;
    dirFlags = ~[files.isdir] ;
    cur_chr_files = files(dirFlags) ;
    for p = 1:length(cur_chr_files)
        file_name = cur_chr_files(p).name ;
        file_path = strcat(chr_dir_path, '\', file_name) ;
        chr_img = imread(file_path) ;
        for cur_size = 1:l_size
            for r = 1:cur_size
                sz = size(chr_img) ;
                background = bg(1:sz(1), 1:cur_size*sz(2), :) ;
                cols = (1:sz(2))+((r-1)*sz(2)) ;
                background(:, cols, :) = chr_img ;
                sp = split(file_name, '.') ;
                name = sp{1} ;
                format = sp{2} ;
                imwrite(background, [chr_output_dir_path '\' 'r' name '_' num2str(cur_size) '_' num2str(r) '.' format])
            end
            
            if cur_size == 1
                continue
            end
            
            for c = 1:cur_size
                sz = size(chr_img) ;
                background = bg(1:cur_size*sz(1), 1:sz(2), :) ;
                rows = (1:sz(1))+((c-1)*sz(1)) ;
                background(rows, :, :) = chr_img ;
                sp = split(file_name, '.') ;
                name = sp{1} ;
                format = sp{2} ;
                imwrite(background, [chr_output_dir_path '\' 'c' name '_' num2str(cur_size) '_' num2str(c) '.' format])
            end
            
            for r = 1:cur_size
                for c = 1:cur_size
                    background = bg(1:cur_size*sz(1), 1:cur_size*sz(2), :) ;
                    cols = (1:sz(2))+((r-1)*sz(2)) ;
                    rows = (1:sz(1))+((c-1)*sz(1)) ;
                    background(rows, cols, :) = chr_img ;
                    sp = split(file_name, '.') ;
                    name = sp{1} ;
                    format = sp{2} ;
                    imwrite(background, [chr_output_dir_path '\' 'r_c' name '_' num2str(cur_size) '_' num2str(r) '_' num2str(c) '.' format])
                end 
            end
        end
    end
end

end