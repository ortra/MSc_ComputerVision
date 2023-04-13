from utils import get_nof_params
from xcpetion import build_xception_backbone

model = build_xception_backbone()
nof_params = get_nof_params(model)

print('\n\nThe number of Xception params is: {}\n\n'.format(nof_params))