"""
Flow layout implementation for VSAT.

This module provides a PyQt layout that arranges widgets in a flow, similar to text in a paragraph.
"""

import logging
from typing import List, Optional

from PyQt6.QtCore import Qt, QRect, QPoint, QSize
from PyQt6.QtWidgets import QLayout, QLayoutItem, QSizePolicy, QStyle, QWidget

logger = logging.getLogger(__name__)

class FlowLayout(QLayout):
    """Flow layout for arranging widgets in a flowing manner, like text in a paragraph."""
    
    def __init__(self, parent=None, margin=0, spacing=-1):
        """Initialize the flow layout.
        
        Args:
            parent: Parent widget
            margin: Layout margin
            spacing: Layout spacing
        """
        super().__init__(parent)
        
        self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)
        
        self.items = []
    
    def addItem(self, item):
        """Add an item to the layout.
        
        Args:
            item: Layout item to add
        """
        self.items.append(item)
    
    def count(self):
        """Get the number of items in the layout.
        
        Returns:
            int: Number of items
        """
        return len(self.items)
    
    def itemAt(self, index):
        """Get the item at the given index.
        
        Args:
            index: Item index
            
        Returns:
            QLayoutItem: Layout item or None if index is out of range
        """
        if 0 <= index < len(self.items):
            return self.items[index]
        return None
    
    def takeAt(self, index):
        """Remove and return the item at the given index.
        
        Args:
            index: Item index
            
        Returns:
            QLayoutItem: Layout item or None if index is out of range
        """
        if 0 <= index < len(self.items):
            return self.items.pop(index)
        return None
    
    def expandingDirections(self):
        """Get the expanding directions of the layout.
        
        Returns:
            Qt.Orientations: Expanding directions
        """
        return Qt.Orientation(0)
    
    def hasHeightForWidth(self):
        """Check if the layout has a height for width.
        
        Returns:
            bool: True if the layout has a height for width
        """
        return True
    
    def heightForWidth(self, width):
        """Get the height for the given width.
        
        Args:
            width: Width
            
        Returns:
            int: Height
        """
        return self.doLayout(QRect(0, 0, width, 0), True)
    
    def setGeometry(self, rect):
        """Set the geometry of the layout.
        
        Args:
            rect: Layout rectangle
        """
        super().setGeometry(rect)
        self.doLayout(rect, False)
    
    def sizeHint(self):
        """Get the size hint of the layout.
        
        Returns:
            QSize: Size hint
        """
        return QSize(100, 100)
    
    def minimumSize(self):
        """Get the minimum size of the layout.
        
        Returns:
            QSize: Minimum size
        """
        size = QSize(0, 0)
        for item in self.items:
            size = size.expandedTo(item.minimumSize())
        
        margin = self.contentsMargins()
        size += QSize(margin.left() + margin.right(), margin.top() + margin.bottom())
        return size
    
    def doLayout(self, rect, testOnly):
        """Perform the layout.
        
        Args:
            rect: Layout rectangle
            testOnly: Whether to only test the layout
            
        Returns:
            int: Height
        """
        x = rect.x()
        y = rect.y()
        lineHeight = 0
        
        for item in self.items:
            widget = item.widget()
            spaceX = self.spacing() + widget.style().layoutSpacing(
                QSizePolicy.ControlType.PushButton,
                QSizePolicy.ControlType.PushButton,
                Qt.Orientation.Horizontal
            )
            spaceY = self.spacing() + widget.style().layoutSpacing(
                QSizePolicy.ControlType.PushButton,
                QSizePolicy.ControlType.PushButton,
                Qt.Orientation.Vertical
            )
            
            nextX = x + item.sizeHint().width() + spaceX
            if nextX - spaceX > rect.right() and lineHeight > 0:
                x = rect.x()
                y = y + lineHeight + spaceY
                nextX = x + item.sizeHint().width() + spaceX
                lineHeight = 0
            
            if not testOnly:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))
            
            x = nextX
            lineHeight = max(lineHeight, item.sizeHint().height())
        
        return y + lineHeight - rect.y() 