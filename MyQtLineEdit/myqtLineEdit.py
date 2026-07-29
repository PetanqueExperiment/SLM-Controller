from pyqtgraph.Qt import QtWidgets
from PyQt5.QtCore import Qt

class MyQtLineEdit(QtWidgets.QLineEdit):
    '''
    This customized lineEdit reacts to Key_Up (Key_Down) event and increment (decrement)
    the value contained in the lineEdit
    '''
    def __init__(self,args):
        QtWidgets.QLineEdit.__init__(self, args)
    
    def keyPressEvent(self, event):
        super(MyQtLineEdit, self).keyPressEvent(event)
        
        if event.key() == Qt.Key_Up:
            self.increment(1)
        if event.key() == Qt.Key_Down:
            self.increment(-1)
    
    def increment(self, sign):
        
        text = self.text()
        value = float(text)
        cpos = self.cursorPosition()
        
        if cpos == 0: 
            #there is no integer before the cursor
            return 
        
        if text[cpos-1].isnumeric():
            
            decimalpos = text.find('.') #find if there is a decimal point
            
            if decimalpos == -1: #we are dealing with an integer number
                power = len(text)-cpos
                precision = 0
                
                
            else: #we are dealing with a floating number
                
                if cpos <= decimalpos:
                    power = decimalpos-cpos
                else:
                    power = decimalpos-cpos+1
                
                precision = len(text)-decimalpos-1#number of digit after decimal point..
            
            value += sign*10**power
            newtext = f'{value:.{precision}f}'
            self.setText(newtext)
            
            newcpos = cpos + len(newtext) - len(text)
            self.setCursorPosition(newcpos)
