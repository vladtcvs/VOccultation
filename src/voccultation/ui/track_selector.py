#
# Copyright (c) 2026 Vladislav Tsendrovskii
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#

import wx
import wx.lib.newevent

TrackSelectedEventType, EVT_TRACK_SELECTED = wx.lib.newevent.NewCommandEvent()

class TrackSelectedEvent(wx.PyCommandEvent):
    eventType = TrackSelectedEventType
    def __init__(self, windowID, track):
        wx.PyCommandEvent(self, self.eventType, windowID)
        self.SetEventObject(track)

TrackRemoveClickedEventType, EVT_TRACK_REMOVE_CLICKED = wx.lib.newevent.NewCommandEvent()
class TrackRemoveClickedEvent(wx.PyCommandEvent):
    eventType = TrackRemoveClickedEventType
    def __init__(self, windowID, track):
        wx.PyCommandEvent(self, self.eventType, windowID)
        self.SetEventObject(track)

NewTrackClickedEventType, EVT_NEW_TRACK_CLICKED = wx.lib.newevent.NewCommandEvent()
class TrackRemoveClickedEvent(wx.PyCommandEvent):
    eventType = NewTrackClickedEventType
    def __init__(self, windowID):
        wx.PyCommandEvent(self, self.eventType, windowID)

class TrackSelectionPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.current_idx = None
        self.objects = []

    def clear(self):
        self.objects = []

    def get_current_track(self) -> object:
        pass

    def add_new_track(self, object, is_removable : bool):
        pass

    def remove_track(self, object):
        pass


    def on_new_track_clicked(self, evt):
        wx.PostEvent()