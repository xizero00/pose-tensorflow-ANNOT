function annolist = util_set_scale(annolist,ref_height)
% 给annotlist中加入计算的人的高度所对应的尺度
if (nargin < 2)
    % reference height in px
    ref_height = 200;
end

% 难道是头部大小占人高度的比例是1/8?
HEAD_HEIGHT_RATIO = 1/8;

for imgidx = 1:length(annolist)
    if (isfield(annolist(imgidx), 'annorect') && ~isempty(annolist(imgidx).annorect))
        rect = annolist(imgidx).annorect;
        for ridx = 1:length(rect)
            if (isfield(rect(ridx), 'annopoints') && ~isempty(rect(ridx).annopoints))
                headSize = util_get_head_size(rect(ridx));
                % 尺度
                % scale = 头部大小*8/参考高度(200)
                % 头部大小*8就是人的高度，然后除以参考高度就是尺度
                sc = ref_height*HEAD_HEIGHT_RATIO/headSize;
                % 尺度应该在0到100之间
                assert(sc<100 && sc>0.01);
                rect(ridx).scale = 1/double(sc);
            end
        end
        annolist(imgidx).annorect = rect;
    end
end