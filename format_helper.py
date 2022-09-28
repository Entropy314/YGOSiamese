def formalize(filename): 
    filename.replace(' ', '_').replace('.','___').replace(':', '--').replace('!', '---').replace('?', '_____').replace('"', '______').replace('★','SSTTAARR')
    return filename

def undo_formalize(filename):
    filename = filename.replace('SSTTAARR','★').replace('______', '"').replace('_____', '?').replace('---', '!').replace('--', ':').replace('___', '.').replace('_', ' ')
    return filename
    
mapping = {' ': '_','.':'___', ':':'--', '!':'---', '?': '_____', '★':'SSTTAARR'}