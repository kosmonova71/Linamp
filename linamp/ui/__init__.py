"""Shared GTK helpers for the LinAmp UI package."""

from __future__ import annotations

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk


def _create_labeled_icon_button(text: str, icon_name: str) -> Gtk.Button:
    """Build a button showing a symbolic icon next to a text label.

    The returned widget is a plain ``Gtk.Button`` so callers can connect
    ``"clicked"``, set tooltips and add their own CSS classes to it.
    """
    button = Gtk.Button()
    button.add_css_class("labeled-icon-button")
    button.set_tooltip_text(text)

    content = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
    content.set_halign(Gtk.Align.CENTER)
    content.set_valign(Gtk.Align.CENTER)

    icon = Gtk.Image.new_from_icon_name(icon_name)
    icon.add_css_class("btn-icon")
    content.append(icon)

    label = Gtk.Label(label=text)
    label.add_css_class("btn-text")
    content.append(label)

    button.set_child(content)
    return button


def _create_modal_form(
    title: str,
    parent: Gtk.Window | None,
    field_labels: list[str],
    action_label: str,
) -> tuple[Gtk.Dialog, list[Gtk.Entry], Gtk.Widget, Gtk.Widget]:
    """Build a modal dialog with one text entry per label.

    Returns the dialog, the entries in the order of ``field_labels``, the
    cancel button and the action button.

    The buttons are typed as ``Gtk.Widget`` because that is the type
    returned by :meth:`Gtk.Dialog.add_button` per the GTK4 API — in
    practice they are :class:`Gtk.Button` instances, but the type
    signature of ``add_button`` only exposes the common base class.
    """
    dialog = Gtk.Dialog(title=title, transient_for=parent, modal=True)
    content = dialog.get_content_area()
    content.set_spacing(8)
    content.set_margin_top(12)
    content.set_margin_bottom(12)
    content.set_margin_start(12)
    content.set_margin_end(12)

    entries: list[Gtk.Entry] = []
    for field_label in field_labels:
        field_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        label = Gtk.Label(label=field_label)
        label.set_halign(Gtk.Align.START)
        field_box.append(label)
        entry = Gtk.Entry()
        entry.set_activates_default(True)
        field_box.append(entry)
        content.append(field_box)
        entries.append(entry)

    cancel_button = dialog.add_button("Cancel", Gtk.ResponseType.CANCEL)
    action_button = dialog.add_button(action_label, Gtk.ResponseType.OK)
    action_button.add_css_class("suggested-action")
    dialog.set_default_response(Gtk.ResponseType.OK)
    return dialog, entries, cancel_button, action_button


__all__ = ["_create_labeled_icon_button", "_create_modal_form"]
