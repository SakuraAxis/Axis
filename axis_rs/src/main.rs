fn main() {
    println!("||||||  Axis GPU Diagnostics  ||||||");

    let instance = wgpu::Instance::default();
    let adapters = instance.enumerate_adapters(wgpu::Backends::all());

    if adapters.is_empty() {
        println!("No WebGPU adapters found.");
        return;
    }

    for (i, adapter) in adapters.iter().enumerate() {
        let info = adapter.get_info();
        println!("[Adapter {}]", i);
        println!("  Name: {}", info.name);
        println!("  Backend: {:?}", info.backend);
        println!("  Driver: {}", info.driver);
        println!();
    }
}