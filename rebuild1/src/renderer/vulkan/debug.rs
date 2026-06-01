use std::ffi::{CStr, c_char, c_void};

use ash::{Entry, Instance, ext, vk};

use super::VulkanError;

const WANT_VALIDATION: bool = cfg!(debug_assertions);
const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

pub(super) struct VulkanDebugConfig {
    validation_layer_enabled: bool,
    debug_utils_enabled: bool,
}

impl VulkanDebugConfig {
    /// Detects optional Vulkan validation support for this process.
    pub(super) fn new(entry: &Entry) -> Result<Self, VulkanError> {
        if !WANT_VALIDATION {
            tracing::trace!("Vulkan validation disabled for non-debug build");
            return Ok(Self {
                validation_layer_enabled: false,
                debug_utils_enabled: false,
            });
        }

        let validation_layer_enabled = instance_layer_supported(entry, VALIDATION_LAYER)?;
        let debug_utils_enabled = instance_extension_supported(entry, ext::debug_utils::NAME)?;

        if validation_layer_enabled {
            tracing::info!("Vulkan validation layer enabled");
        } else {
            tracing::warn!(
                layer = %VALIDATION_LAYER.to_string_lossy(),
                "Vulkan validation layer is unavailable"
            );
        }

        if debug_utils_enabled {
            tracing::info!("Vulkan debug utils enabled");
        } else {
            tracing::warn!("Vulkan debug utils extension is unavailable");
        }

        Ok(Self {
            validation_layer_enabled,
            debug_utils_enabled,
        })
    }

    /// Returns whether validation layer support was detected and enabled.
    pub(super) fn validation_layer_enabled(&self) -> bool {
        self.validation_layer_enabled
    }

    /// Returns whether debug utils support was detected and enabled.
    pub(super) fn debug_utils_enabled(&self) -> bool {
        self.debug_utils_enabled
    }

    /// Adds optional debug instance extensions to the required extension list.
    pub(super) fn append_extensions(&self, extensions: &mut Vec<*const c_char>) {
        if self.debug_utils_enabled {
            extensions.push(ext::debug_utils::NAME.as_ptr());
        }
    }

    /// Returns instance layer names that should be enabled.
    pub(super) fn layer_names(&self) -> Vec<*const c_char> {
        if self.validation_layer_enabled {
            vec![VALIDATION_LAYER.as_ptr()]
        } else {
            Vec::new()
        }
    }
}

pub(super) struct VulkanDebug {
    loader: ext::debug_utils::Instance,
    messenger: vk::DebugUtilsMessengerEXT,
}

impl VulkanDebug {
    /// Creates the Vulkan debug messenger when debug utils are enabled.
    pub(super) fn create(
        entry: &Entry,
        instance: &Instance,
        enabled: bool,
    ) -> Result<Option<Self>, VulkanError> {
        if !enabled {
            return Ok(None);
        }

        let loader = ext::debug_utils::Instance::new(entry, instance);
        let create_info = messenger_create_info();
        let messenger = create_debug_messenger(&loader, &create_info)?;
        tracing::info!("created Vulkan debug messenger");

        Ok(Some(Self { loader, messenger }))
    }

    /// Destroys the debug messenger before its parent instance is destroyed.
    pub(super) fn destroy(self) {
        tracing::trace!("destroying Vulkan debug messenger");
        destroy_debug_messenger(&self.loader, self.messenger);
    }
}

/// Builds the callback configuration used during instance creation and messenger creation.
pub(super) fn messenger_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT<'static> {
    vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(vulkan_debug_callback))
}

/// Returns whether the requested Vulkan instance layer is available.
fn instance_layer_supported(entry: &Entry, layer_name: &CStr) -> Result<bool, VulkanError> {
    let layers = enumerate_instance_layers(entry)?;
    Ok(layers
        .iter()
        .any(|layer| layer_name_matches(layer, layer_name)))
}

/// Returns whether the requested Vulkan instance extension is available.
fn instance_extension_supported(entry: &Entry, extension_name: &CStr) -> Result<bool, VulkanError> {
    let extensions = enumerate_instance_extensions(entry)?;
    Ok(extensions
        .iter()
        .any(|extension| extension_name_matches(extension, extension_name)))
}

/// Creates the debug messenger used by the validation layer callback.
fn create_debug_messenger(
    loader: &ext::debug_utils::Instance,
    create_info: &vk::DebugUtilsMessengerCreateInfoEXT<'_>,
) -> Result<vk::DebugUtilsMessengerEXT, VulkanError> {
    // Safety: debug utils is enabled on the instance, and `create_info` only references a static
    // callback function with no custom user data.
    unsafe { loader.create_debug_utils_messenger(create_info, None) }.map_err(VulkanError::Vk)
}

/// Destroys the debug messenger owned by one Vulkan context.
fn destroy_debug_messenger(
    loader: &ext::debug_utils::Instance,
    messenger: vk::DebugUtilsMessengerEXT,
) {
    // Safety: the messenger was created by this loader and is destroyed exactly once before the
    // parent instance is destroyed.
    unsafe {
        loader.destroy_debug_utils_messenger(messenger, None);
    }
}

/// Receives Vulkan debug messages and forwards them into tracing.
unsafe extern "system" fn vulkan_debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut c_void,
) -> vk::Bool32 {
    let message = vulkan_debug_message(data);

    match severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
            tracing::error!(message_type = ?message_type, message = %message, "Vulkan debug message");
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => {
            tracing::warn!(message_type = ?message_type, message = %message, "Vulkan debug message");
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => {
            tracing::trace!(message_type = ?message_type, message = %message, "Vulkan debug message");
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => {
            tracing::trace!(message_type = ?message_type, message = %message, "Vulkan debug message");
        }
        _ => {
            tracing::trace!(message_type = ?message_type, message = %message, "Vulkan debug message");
        }
    }

    vk::FALSE
}

/// Copies the Vulkan debug callback message out of the raw callback data.
fn vulkan_debug_message(data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>) -> String {
    if data.is_null() {
        return "<missing Vulkan debug callback data>".to_owned();
    }

    // Safety: Vulkan passes a valid callback data pointer for the duration of the callback.
    let message = unsafe { (*data).p_message };
    if message.is_null() {
        return "<missing Vulkan debug callback message>".to_owned();
    }

    // Safety: Vulkan guarantees that `p_message` is a null-terminated string.
    unsafe { CStr::from_ptr(message) }
        .to_string_lossy()
        .into_owned()
}

/// Compares a Vulkan layer property name with a required static layer name.
fn layer_name_matches(property: &vk::LayerProperties, expected: &CStr) -> bool {
    // Safety: Vulkan guarantees that `layer_name` is a null-terminated C string.
    (unsafe { CStr::from_ptr(property.layer_name.as_ptr()) }) == expected
}

/// Compares a Vulkan extension property name with a required static extension name.
fn extension_name_matches(property: &vk::ExtensionProperties, expected: &CStr) -> bool {
    // Safety: Vulkan guarantees that `extension_name` is a null-terminated C string.
    (unsafe { CStr::from_ptr(property.extension_name.as_ptr()) }) == expected
}

/// Enumerates Vulkan instance layers exposed by the active loader.
fn enumerate_instance_layers(entry: &Entry) -> Result<Vec<vk::LayerProperties>, VulkanError> {
    // Safety: `entry` owns the loaded Vulkan entry points and no returned pointers escape.
    unsafe { entry.enumerate_instance_layer_properties() }.map_err(VulkanError::Vk)
}

/// Enumerates Vulkan instance extensions exposed by the active loader.
fn enumerate_instance_extensions(
    entry: &Entry,
) -> Result<Vec<vk::ExtensionProperties>, VulkanError> {
    // Safety: `entry` owns the loaded Vulkan entry points and no returned pointers escape.
    unsafe { entry.enumerate_instance_extension_properties(None) }.map_err(VulkanError::Vk)
}
